package net.finmath.finitedifference.solvers.adi;

import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

import net.finmath.finitedifference.FiniteDifferenceExerciseUtil;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMHestonModel;
import net.finmath.finitedifference.assetderivativevaluation.products.FiniteDifferenceInternalStateConstraint;
import net.finmath.finitedifference.assetderivativevaluation.products.FiniteDifferenceProduct;
import net.finmath.finitedifference.boundaries.BoundaryCondition;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.solvers.FDMSolver;
import net.finmath.finitedifference.solvers.ThomasSolver;
import net.finmath.finitedifference.solvers.TridiagonalMatrix;
import net.finmath.modelling.Exercise;

/**
 * Stabilization-first ADI solver for 2D Heston PDEs on state variables (S, v).
 *
 * <p>
 * This implementation uses a conservative Douglas-style ADI step with two half-substeps
 * per PDE time step.
 * </p>
 *
 * <p>
 * Operator split:
 * </p>
 * <ul>
 *   <li>A0: mixed derivative + discount term (explicit),</li>
 *   <li>A1: spot-direction drift + diffusion (implicit line solve),</li>
 *   <li>A2: variance-direction drift + diffusion (implicit line solve).</li>
 * </ul>
 *
 * <p>
 * Flattening convention:
 * {@code k = iS + iV * nS}, where {@code iS} is the fastest index.
 * </p>
 *
 * <p>
 * Obstacle handling:
 * For American and Bermudan exercise, this solver applies a post-step projection
 * {@code u = max(u, payoff)} whenever exercise is allowed at the current running time.
 * Internal state constraints (e.g. barriers) remain hard constraints and take precedence
 * over the exercise obstacle.
 * </p>
 */
public class FDMHestonADI2D implements FDMSolver {

	private final FDMHestonModel model;
	private final FiniteDifferenceProduct product;
	private final SpaceTimeDiscretization spaceTimeDiscretization;
	private final Exercise exercise;

	private final double theta;

	private final double[] sGrid;
	private final double[] vGrid;

	private final int nS;
	private final int nV;
	private final int n;

	private final HestonADIStencilBuilder stencilBuilder;

	public FDMHestonADI2D(
			final FDMHestonModel model,
			final FiniteDifferenceProduct product,
			final SpaceTimeDiscretization spaceTimeDiscretization,
			final Exercise exercise) {

		this.model = model;
		this.product = product;
		this.spaceTimeDiscretization = spaceTimeDiscretization;
		this.exercise = exercise;

		final Grid sGridObj = spaceTimeDiscretization.getSpaceGrid(0);
		final Grid vGridObj = spaceTimeDiscretization.getSpaceGrid(1);
		if(sGridObj == null || vGridObj == null) {
			throw new IllegalArgumentException("FDMHestonADI2D requires a 2D discretization.");
		}

		this.sGrid = sGridObj.getGrid();
		this.vGrid = vGridObj.getGrid();

		this.nS = sGrid.length;
		this.nV = vGrid.length;
		this.n = nS * nV;

		this.theta = Math.max(0.5, spaceTimeDiscretization.getTheta());
		this.stencilBuilder = new HestonADIStencilBuilder(model, sGrid, vGrid);
	}

	@Override
	public double[][] getValues(final double time, final DoubleUnaryOperator valueAtMaturity) {
		return getValues(time, (s, v) -> valueAtMaturity.applyAsDouble(s));
	}

	public double[][] getValues(final double time, final DoubleBinaryOperator valueAtMaturity) {

		final int timeLength = spaceTimeDiscretization.getTimeDiscretization().getNumberOfTimeSteps() + 1;
		final int numberOfTimeSteps = spaceTimeDiscretization.getTimeDiscretization().getNumberOfTimeSteps();

		double[] u = new double[n];
		for(int j = 0; j < nV; j++) {
			for(int i = 0; i < nS; i++) {
				u[flatten(i, j)] = valueAtMaturity.applyAsDouble(sGrid[i], vGrid[j]);
			}
		}

		applyOuterBoundaries(time, u);
		applyInternalConstraints(time, u);
		u = sanitize(u);

		final RealMatrix solutionSurface = new Array2DRowRealMatrix(n, timeLength);
		solutionSurface.setColumn(0, u.clone());

		for(int m = 0; m < numberOfTimeSteps; m++) {
			final double dt = spaceTimeDiscretization.getTimeDiscretization().getTimeStep(m);

			final double tauNext = spaceTimeDiscretization.getTimeDiscretization().getTime(m + 1);
			final double runningTimeNext =
					spaceTimeDiscretization.getTimeDiscretization().getLastTime() - tauNext;

			u = performStableDouglasStep(u, runningTimeNext, dt);

			/*
			 * End-of-step enforcement:
			 * 1) internal hard constraints
			 * 2) outer boundaries
			 * 3) early-exercise obstacle, if allowed
			 * 4) restore hard constraints / boundaries after projection
			 */
			applyInternalConstraints(runningTimeNext, u);
			applyOuterBoundaries(runningTimeNext, u);

			applyExerciseObstacleIfNeeded(runningTimeNext, tauNext, u, valueAtMaturity);

			applyInternalConstraints(runningTimeNext, u);
			applyOuterBoundaries(runningTimeNext, u);

			u = sanitize(u);

			solutionSurface.setColumn(m + 1, u.clone());
		}

		return solutionSurface.getData();
	}

	@Override
	public double[] getValue(
			final double evaluationTime,
			final double time,
			final DoubleUnaryOperator valueAtMaturity) {

		final RealMatrix values = new Array2DRowRealMatrix(getValues(time, valueAtMaturity));
		final double tau = time - evaluationTime;
		final int timeIndex = spaceTimeDiscretization.getTimeDiscretization().getTimeIndexNearestLessOrEqual(tau);
		return values.getColumn(timeIndex);
	}

	public double[] getValue(
			final double evaluationTime,
			final double time,
			final DoubleBinaryOperator valueAtMaturity) {

		final RealMatrix values = new Array2DRowRealMatrix(getValues(time, valueAtMaturity));
		final double tau = time - evaluationTime;
		final int timeIndex = spaceTimeDiscretization.getTimeDiscretization().getTimeIndexNearestLessOrEqual(tau);
		return values.getColumn(timeIndex);
	}

	/**
	 * Conservative stabilization-first step:
	 * split one PDE step into two half Douglas ADI steps.
	 */
	protected double[] performStableDouglasStep(
			final double[] u,
			final double currentTime,
			final double dt) {

		final double halfDt = 0.5 * dt;

		double[] uMid = performDouglasHalfStep(u, currentTime + halfDt, halfDt);
		uMid = sanitize(uMid);

		double[] uNext = performDouglasHalfStep(uMid, currentTime, halfDt);
		uNext = sanitize(uNext);

		return uNext;
	}

	protected double[] performDouglasHalfStep(
			final double[] u,
			final double currentTime,
			final double dt) {

		final double[] explicit = applyFullExplicitOperator(u, currentTime);
		final double[] y0 = add(u, scale(explicit, dt));

		/*
		 * During a half-step, keep outer boundaries consistent, but do not clamp the
		 * internal constraint or early-exercise obstacle yet.
		 */
		applyOuterBoundaries(currentTime, y0);

		final double[] a1u = applyA1Explicit(u, currentTime);
		final double[] rhs1 = subtract(y0, scale(a1u, theta * dt));
		double[] y1 = solveSpotLines(rhs1, currentTime, dt);
		y1 = sanitize(y1);

		applyOuterBoundaries(currentTime, y1);

		final double[] a2u = applyA2Explicit(u, currentTime);
		final double[] rhs2 = subtract(y1, scale(a2u, theta * dt));
		double[] y2 = solveVarianceLines(rhs2, currentTime, dt);
		y2 = sanitize(y2);

		/*
		 * Only now, at the end of the half-step, enforce internal hard constraints.
		 * Early exercise is still deferred to the completed full step.
		 */
		applyInternalConstraints(currentTime, y2);
		applyOuterBoundaries(currentTime, y2);

		return y2;
	}

	protected double[] applyFullExplicitOperator(final double[] u, final double time) {
		return add(add(applyA0Explicit(u, time), applyA1Explicit(u, time)), applyA2Explicit(u, time));
	}

	/**
	 * Explicit mixed-derivative plus discount operator.
	 */
	protected double[] applyA0Explicit(final double[] u, final double time) {

		final double[] out = new double[n];

		final double tSafe = Math.max(time, 1E-10);
		final double discountFactor = model.getRiskFreeCurve().getDiscountFactor(tSafe);
		final double r = -Math.log(discountFactor) / tSafe;

		for(int j = 1; j < nV - 1; j++) {
			for(int i = 1; i < nS - 1; i++) {
				final int k = flatten(i, j);

				final double s = sGrid[i];
				final double v = vGrid[j];

				final double[][] b = model.getFactorLoading(time, s, v);

				double aSV = 0.0;
				for(int f = 0; f < b[0].length; f++) {
					aSV += b[0][f] * b[1][f];
				}

				final double dsDown = sGrid[i] - sGrid[i - 1];
				final double dsUp = sGrid[i + 1] - sGrid[i];
				final double dvDown = vGrid[j] - vGrid[j - 1];
				final double dvUp = vGrid[j + 1] - vGrid[j];

				final double dSdV =
						(
								u[flatten(i + 1, j + 1)]
								- u[flatten(i + 1, j - 1)]
								- u[flatten(i - 1, j + 1)]
								+ u[flatten(i - 1, j - 1)]
						)
						/ ((dsDown + dsUp) * (dvDown + dvUp));

				out[k] = aSV * dSdV - r * u[k];
			}
		}

		return out;
	}

	/**
	 * Explicit spot-direction operator.
	 */
	protected double[] applyA1Explicit(final double[] u, final double time) {

		final double[] out = new double[n];

		for(int j = 0; j < nV; j++) {
			for(int i = 1; i < nS - 1; i++) {
				final int k = flatten(i, j);

				final double s = sGrid[i];
				final double v = vGrid[j];

				final double[] drift = model.getDrift(time, s, v);
				final double[][] b = model.getFactorLoading(time, s, v);

				final double muS = drift[0];

				double aSS = 0.0;
				for(int f = 0; f < b[0].length; f++) {
					aSS += b[0][f] * b[0][f];
				}

				final double dsDown = sGrid[i] - sGrid[i - 1];
				final double dsUp = sGrid[i + 1] - sGrid[i];

				final double dS =
						(u[flatten(i + 1, j)] - u[flatten(i - 1, j)])
						/ (dsDown + dsUp);

				final double dSS =
						2.0 * (
								(u[flatten(i + 1, j)] - u[k]) / dsUp
								- (u[k] - u[flatten(i - 1, j)]) / dsDown
						)
						/ (dsDown + dsUp);

				out[k] = muS * dS + 0.5 * aSS * dSS;
			}
		}

		return out;
	}

	/**
	 * Explicit variance-direction operator.
	 */
	protected double[] applyA2Explicit(final double[] u, final double time) {

		final double[] out = new double[n];

		for(int j = 1; j < nV - 1; j++) {
			for(int i = 0; i < nS; i++) {
				final int k = flatten(i, j);

				final double s = sGrid[i];
				final double v = vGrid[j];

				final double[] drift = model.getDrift(time, s, v);
				final double[][] b = model.getFactorLoading(time, s, v);

				final double muV = drift[1];

				double aVV = 0.0;
				for(int f = 0; f < b[1].length; f++) {
					aVV += b[1][f] * b[1][f];
				}

				final double dvDown = vGrid[j] - vGrid[j - 1];
				final double dvUp = vGrid[j + 1] - vGrid[j];

				final double dV =
						(u[flatten(i, j + 1)] - u[flatten(i, j - 1)])
						/ (dvDown + dvUp);

				final double dVV =
						2.0 * (
								(u[flatten(i, j + 1)] - u[k]) / dvUp
								- (u[k] - u[flatten(i, j - 1)]) / dvDown
						)
						/ (dvDown + dvUp);

				out[k] = muV * dV + 0.5 * aVV * dVV;
			}
		}

		return out;
	}

	/**
	 * Implicit solve along spot lines for fixed variance.
	 */
	protected double[] solveSpotLines(
			final double[] rhs,
			final double time,
			final double dt) {

		final double[] out = rhs.clone();

		for(int j = 0; j < nV; j++) {
			final TridiagonalMatrix m = stencilBuilder.buildSpotLineMatrix(time, dt, theta, j);

			final double[] lineRhs = new double[nS];
			for(int i = 0; i < nS; i++) {
				lineRhs[i] = rhs[flatten(i, j)];
			}

			final double lowerBoundaryValue = getLowerBoundaryValueForSpot(time, j, lineRhs[0]);
			final double upperBoundaryValue = getUpperBoundaryValueForSpot(time, j, lineRhs[nS - 1]);

			overwriteBoundaryRow(m, lineRhs, 0, lowerBoundaryValue);
			overwriteBoundaryRow(m, lineRhs, nS - 1, upperBoundaryValue);

			final double[] solved = ThomasSolver.solve(m.lower, m.diag, m.upper, lineRhs);

			for(int i = 0; i < nS; i++) {
				out[flatten(i, j)] = solved[i];
			}
		}

		return out;
	}

	/**
	 * Implicit solve along variance lines for fixed spot.
	 */
	protected double[] solveVarianceLines(
			final double[] rhs,
			final double time,
			final double dt) {

		final double[] out = rhs.clone();

		for(int i = 0; i < nS; i++) {
			final TridiagonalMatrix m = stencilBuilder.buildVarianceLineMatrix(time, dt, theta, i);

			final double[] lineRhs = new double[nV];
			for(int j = 0; j < nV; j++) {
				lineRhs[j] = rhs[flatten(i, j)];
			}

			final double lowerBoundaryValue = getLowerBoundaryValueForVariance(time, i, lineRhs[0]);
			final double upperBoundaryValue = getUpperBoundaryValueForVariance(time, i, lineRhs[nV - 1]);

			overwriteBoundaryRow(m, lineRhs, 0, lowerBoundaryValue);
			overwriteBoundaryRow(m, lineRhs, nV - 1, upperBoundaryValue);

			final double[] solved = ThomasSolver.solve(m.lower, m.diag, m.upper, lineRhs);

			for(int j = 0; j < nV; j++) {
				out[flatten(i, j)] = solved[j];
			}
		}

		return out;
	}

	protected void applyOuterBoundaries(final double time, final double[] u) {

		for(int j = 0; j < nV; j++) {
			u[flatten(0, j)] = getLowerBoundaryValueForSpot(time, j, u[flatten(0, j)]);
			u[flatten(nS - 1, j)] = getUpperBoundaryValueForSpot(time, j, u[flatten(nS - 1, j)]);
		}

		for(int i = 0; i < nS; i++) {
			u[flatten(i, 0)] = getLowerBoundaryValueForVariance(time, i, u[flatten(i, 0)]);
			u[flatten(i, nV - 1)] = getUpperBoundaryValueForVariance(time, i, u[flatten(i, nV - 1)]);
		}
	}

	protected void applyInternalConstraints(final double time, final double[] u) {
		if(!(product instanceof FiniteDifferenceInternalStateConstraint)) {
			return;
		}

		final FiniteDifferenceInternalStateConstraint constraint =
				(FiniteDifferenceInternalStateConstraint) product;

		for(int j = 0; j < nV; j++) {
			for(int i = 0; i < nS; i++) {
				final int k = flatten(i, j);
				if(constraint.isConstraintActive(time, sGrid[i], vGrid[j])) {
					u[k] = constraint.getConstrainedValue(time, sGrid[i], vGrid[j]);
				}
			}
		}
	}

	protected void applyExerciseObstacleIfNeeded(
			final double runningTime,
			final double tau,
			final double[] u,
			final DoubleBinaryOperator valueAtMaturity) {

		final boolean isExerciseAllowed =
				FiniteDifferenceExerciseUtil.isExerciseAllowedAtTimeToMaturity(tau, exercise);

		if(!isExerciseAllowed) {
			return;
		}

		for(int j = 0; j < nV; j++) {
			for(int i = 0; i < nS; i++) {
				if(isInternalConstraintActive(runningTime, sGrid[i], vGrid[j])) {
					continue;
				}

				final int k = flatten(i, j);
				final double payoff = valueAtMaturity.applyAsDouble(sGrid[i], vGrid[j]);
				u[k] = Math.max(u[k], payoff);
			}
		}
	}

	protected boolean isInternalConstraintActive(final double time, final double s, final double v) {
		if(product instanceof FiniteDifferenceInternalStateConstraint) {
			return ((FiniteDifferenceInternalStateConstraint) product).isConstraintActive(time, s, v);
		}
		return false;
	}

	private double getLowerBoundaryValueForSpot(final double time, final int varianceIndex, final double fallback) {
		final BoundaryCondition[] conditions =
				model.getBoundaryConditionsAtLowerBoundary(product, time, sGrid[0], vGrid[varianceIndex]);
		return extractBoundaryValue(conditions[0], fallback);
	}

	private double getUpperBoundaryValueForSpot(final double time, final int varianceIndex, final double fallback) {
		final BoundaryCondition[] conditions =
				model.getBoundaryConditionsAtUpperBoundary(product, time, sGrid[nS - 1], vGrid[varianceIndex]);
		return extractBoundaryValue(conditions[0], fallback);
	}

	private double getLowerBoundaryValueForVariance(final double time, final int spotIndex, final double fallback) {
		final BoundaryCondition[] conditions =
				model.getBoundaryConditionsAtLowerBoundary(product, time, sGrid[spotIndex], vGrid[0]);
		return extractBoundaryValue(conditions[1], fallback);
	}

	private double getUpperBoundaryValueForVariance(final double time, final int spotIndex, final double fallback) {
		final BoundaryCondition[] conditions =
				model.getBoundaryConditionsAtUpperBoundary(product, time, sGrid[spotIndex], vGrid[nV - 1]);
		return extractBoundaryValue(conditions[1], fallback);
	}

	private double extractBoundaryValue(final BoundaryCondition condition, final double fallback) {
		if(condition != null && condition.isDirichlet()) {
			return condition.getValue();
		}
		return fallback;
	}

	private void overwriteBoundaryRow(
			final TridiagonalMatrix m,
			final double[] rhs,
			final int row,
			final double value) {

		m.lower[row] = 0.0;
		m.diag[row] = 1.0;
		m.upper[row] = 0.0;
		rhs[row] = value;
	}

	private int flatten(final int iS, final int iV) {
		return iS + iV * nS;
	}

	private double[] add(final double[] a, final double[] b) {
		final double[] out = new double[a.length];
		for(int i = 0; i < a.length; i++) {
			out[i] = a[i] + b[i];
		}
		return out;
	}

	private double[] subtract(final double[] a, final double[] b) {
		final double[] out = new double[a.length];
		for(int i = 0; i < a.length; i++) {
			out[i] = a[i] - b[i];
		}
		return out;
	}

	private double[] scale(final double[] a, final double c) {
		final double[] out = new double[a.length];
		for(int i = 0; i < a.length; i++) {
			out[i] = c * a[i];
		}
		return out;
	}

	private double[] sanitize(final double[] u) {
		final double[] out = new double[u.length];
		for(int i = 0; i < u.length; i++) {
			final double value = u[i];
			if(!Double.isFinite(value)) {
				out[i] = 0.0;
			}
			else if(value > 1E12) {
				out[i] = 1E12;
			}
			else if(value < -1E12) {
				out[i] = -1E12;
			}
			else {
				out[i] = value;
			}
		}
		return out;
	}
}