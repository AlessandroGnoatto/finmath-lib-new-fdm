package net.finmath.finitedifference.solvers.adi;

import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

import net.finmath.finitedifference.FiniteDifferenceExerciseUtil;
import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
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
 * New ADI base class consuming an explicit semidiscrete split object.
 *
 * <p>
 * This class is intended for the improve-adi-schemes branch and lives in
 * parallel to the legacy {@link AbstractADI2D}. It keeps the existing product
 * and boundary framework, but separates:
 * </p>
 * <ul>
 *   <li>operator construction, handled by {@link ADI2DOperatorSplit},</li>
 *   <li>time stepping, handled by this class.</li>
 * </ul>
 *
 * <p>
 * Supported ADI schemes:
 * </p>
 * <ul>
 *   <li>Douglas</li>
 *   <li>Modified Craig-Sneyd (MCS)</li>
 * </ul>
 */
public abstract class AbstractSplitADI2D implements FDMSolver {

	@FunctionalInterface
	public interface DoubleTernaryOperator {
		double applyAsDouble(double x0, double x1, double x2);
	}

	public enum ADIScheme {
		DOUGLAS,
		MCS
	}

	protected final FiniteDifferenceEquityModel model;
	protected final FiniteDifferenceProduct product;
	protected final SpaceTimeDiscretization spaceTimeDiscretization;
	protected final Exercise exercise;

	protected final ADI2DOperatorSplit operatorSplit;
	protected final ADIScheme adiScheme;

	protected final double theta;

	protected final double[] x0Grid;
	protected final double[] x1Grid;

	protected final int n0;
	protected final int n1;
	protected final int n;

	protected AbstractSplitADI2D(
			final FiniteDifferenceEquityModel model,
			final FiniteDifferenceProduct product,
			final SpaceTimeDiscretization spaceTimeDiscretization,
			final Exercise exercise,
			final ADI2DOperatorSplit operatorSplit) {
		this(model, product, spaceTimeDiscretization, exercise, operatorSplit, ADIScheme.DOUGLAS);
	}

	protected AbstractSplitADI2D(
			final FiniteDifferenceEquityModel model,
			final FiniteDifferenceProduct product,
			final SpaceTimeDiscretization spaceTimeDiscretization,
			final Exercise exercise,
			final ADI2DOperatorSplit operatorSplit,
			final ADIScheme adiScheme) {

		this.model = model;
		this.product = product;
		this.spaceTimeDiscretization = spaceTimeDiscretization;
		this.exercise = exercise;
		this.operatorSplit = operatorSplit;
		this.adiScheme = adiScheme;

		final Grid x0GridObj = spaceTimeDiscretization.getSpaceGrid(0);
		final Grid x1GridObj = spaceTimeDiscretization.getSpaceGrid(1);
		if(x0GridObj == null || x1GridObj == null) {
			throw new IllegalArgumentException("AbstractSplitADI2D requires a 2D discretization.");
		}

		this.x0Grid = x0GridObj.getGrid();
		this.x1Grid = x1GridObj.getGrid();

		this.n0 = x0Grid.length;
		this.n1 = x1Grid.length;
		this.n = n0 * n1;

		this.theta = Math.max(1.0 / 3.0, spaceTimeDiscretization.getTheta());
	}

	@Override
	public double[][] getValues(final double time, final DoubleUnaryOperator valueAtMaturity) {
		return getValues(
				time,
				(x0, x1) -> valueAtMaturity.applyAsDouble(x0),
				(runningTime, x0, x1) -> valueAtMaturity.applyAsDouble(x0));
	}

	public double[][] getValues(final double time, final DoubleBinaryOperator valueAtMaturity) {
		return getValues(
				time,
				valueAtMaturity,
				(runningTime, x0, x1) -> valueAtMaturity.applyAsDouble(x0, x1));
	}

	public double[][] getValues(
			final double time,
			final DoubleBinaryOperator valueAtMaturity,
			final DoubleTernaryOperator exerciseValue) {

		final int timeLength = spaceTimeDiscretization.getTimeDiscretization().getNumberOfTimeSteps() + 1;
		final int numberOfTimeSteps = spaceTimeDiscretization.getTimeDiscretization().getNumberOfTimeSteps();

		double[] u = new double[n];
		for(int j = 0; j < n1; j++) {
			for(int i = 0; i < n0; i++) {
				u[flatten(i, j)] = valueAtMaturity.applyAsDouble(x0Grid[i], x1Grid[j]);
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

			u = performStep(u, runningTimeNext, dt);

			applyInternalConstraints(runningTimeNext, u);
			applyOuterBoundaries(runningTimeNext, u);

			applyExerciseObstacleIfNeeded(runningTimeNext, tauNext, u, exerciseValue);

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

	public double[] getValue(
			final double evaluationTime,
			final double time,
			final DoubleBinaryOperator valueAtMaturity,
			final DoubleTernaryOperator exerciseValue) {

		final RealMatrix values = new Array2DRowRealMatrix(getValues(time, valueAtMaturity, exerciseValue));
		final double tau = time - evaluationTime;
		final int timeIndex = spaceTimeDiscretization.getTimeDiscretization().getTimeIndexNearestLessOrEqual(tau);
		return values.getColumn(timeIndex);
	}

	protected double[] performStep(
			final double[] u,
			final double currentTime,
			final double dt) {

		switch(adiScheme) {
		case MCS:
			return performMCSStep(u, currentTime, dt);
		case DOUGLAS:
		default:
			return performStableDouglasStep(u, currentTime, dt);
		}
	}

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

		final double[] explicit = operatorSplit.applyA(u, currentTime);
		final double[] y0 = add(u, scale(explicit, dt));

		applyOuterBoundaries(currentTime, y0);

		final double[] a1u = operatorSplit.applyA1(u, currentTime);
		final double[] rhs1 = subtract(y0, scale(a1u, theta * dt));
		double[] y1 = solveFirstDirectionLines(rhs1, currentTime, dt);
		y1 = sanitize(y1);

		applyOuterBoundaries(currentTime, y1);

		final double[] a2u = operatorSplit.applyA2(u, currentTime);
		final double[] rhs2 = subtract(y1, scale(a2u, theta * dt));
		double[] y2 = solveSecondDirectionLines(rhs2, currentTime, dt);
		y2 = sanitize(y2);

		applyInternalConstraints(currentTime, y2);
		applyOuterBoundaries(currentTime, y2);

		return y2;
	}

	/**
	 * Modified Craig-Sneyd step.
	 *
	 * <p>
	 * Uses the clean split
	 * </p>
	 * <pre>
	 * A = A0 + A1 + A2
	 * </pre>
	 * <p>
	 * provided by {@link ADI2DOperatorSplit}.
	 * </p>
	 */
	protected double[] performMCSStep(
			final double[] u,
			final double currentTime,
			final double dt) {

		final double oldTime = currentTime + dt;
		final double newTime = currentTime;

		final double[] auOld = operatorSplit.applyA(u, oldTime);

		double[] y0 = add(u, scale(auOld, dt));
		applyOuterBoundaries(newTime, y0);
		applyInternalConstraints(newTime, y0);
		y0 = sanitize(y0);

		final double[] a1uOld = operatorSplit.applyA1(u, oldTime);
		final double[] rhs1 = subtract(y0, scale(a1uOld, theta * dt));
		double[] y1 = solveFirstDirectionLines(rhs1, newTime, dt);
		applyOuterBoundaries(newTime, y1);
		applyInternalConstraints(newTime, y1);
		y1 = sanitize(y1);

		final double[] a2uOld = operatorSplit.applyA2(u, oldTime);
		final double[] rhs2 = subtract(y1, scale(a2uOld, theta * dt));
		double[] y2 = solveSecondDirectionLines(rhs2, newTime, dt);
		applyOuterBoundaries(newTime, y2);
		applyInternalConstraints(newTime, y2);
		y2 = sanitize(y2);

		final double[] a0uOld = operatorSplit.applyA0(u, oldTime);
		final double[] a0y2New = operatorSplit.applyA0(y2, newTime);
		double[] yHat0 = add(y0, scale(subtract(a0y2New, a0uOld), theta * dt));
		applyOuterBoundaries(newTime, yHat0);
		applyInternalConstraints(newTime, yHat0);
		yHat0 = sanitize(yHat0);

		final double[] ay2New = operatorSplit.applyA(y2, newTime);
		double[] yTilde0 = add(yHat0, scale(subtract(ay2New, auOld), (0.5 - theta) * dt));
		applyOuterBoundaries(newTime, yTilde0);
		applyInternalConstraints(newTime, yTilde0);
		yTilde0 = sanitize(yTilde0);

		final double[] rhs1Corr = subtract(yTilde0, scale(a1uOld, theta * dt));
		double[] yTilde1 = solveFirstDirectionLines(rhs1Corr, newTime, dt);
		applyOuterBoundaries(newTime, yTilde1);
		applyInternalConstraints(newTime, yTilde1);
		yTilde1 = sanitize(yTilde1);

		final double[] rhs2Corr = subtract(yTilde1, scale(a2uOld, theta * dt));
		double[] yTilde2 = solveSecondDirectionLines(rhs2Corr, newTime, dt);
		applyOuterBoundaries(newTime, yTilde2);
		applyInternalConstraints(newTime, yTilde2);
		yTilde2 = sanitize(yTilde2);

		return yTilde2;
	}

	protected double[] solveFirstDirectionLines(
			final double[] rhs,
			final double time,
			final double dt) {

		final double[] out = rhs.clone();

		for(int j = 0; j < n1; j++) {
			final TridiagonalMatrix matrix =
					operatorSplit.buildFirstDirectionLineMatrix(time, dt, theta, j);

			final double[] lineRhs = new double[n0];
			for(int i = 0; i < n0; i++) {
				lineRhs[i] = rhs[flatten(i, j)];
			}

			final double lowerBoundaryValue =
					getLowerBoundaryValueForFirstDirection(time, j, lineRhs[0]);
			final double upperBoundaryValue =
					getUpperBoundaryValueForFirstDirection(time, j, lineRhs[n0 - 1]);

			overwriteBoundaryRow(matrix, lineRhs, 0, lowerBoundaryValue);
			overwriteBoundaryRow(matrix, lineRhs, n0 - 1, upperBoundaryValue);

			final double[] solved =
					ThomasSolver.solve(matrix.lower, matrix.diag, matrix.upper, lineRhs);

			for(int i = 0; i < n0; i++) {
				out[flatten(i, j)] = solved[i];
			}
		}

		return out;
	}

	protected double[] solveSecondDirectionLines(
			final double[] rhs,
			final double time,
			final double dt) {

		final double[] out = rhs.clone();

		for(int i = 0; i < n0; i++) {
			final TridiagonalMatrix matrix =
					operatorSplit.buildSecondDirectionLineMatrix(time, dt, theta, i);

			final double[] lineRhs = new double[n1];
			for(int j = 0; j < n1; j++) {
				lineRhs[j] = rhs[flatten(i, j)];
			}

			final double lowerBoundaryValue =
					getLowerBoundaryValueForSecondDirection(time, i, lineRhs[0]);
			final double upperBoundaryValue =
					getUpperBoundaryValueForSecondDirection(time, i, lineRhs[n1 - 1]);

			overwriteBoundaryRow(matrix, lineRhs, 0, lowerBoundaryValue);
			overwriteBoundaryRow(matrix, lineRhs, n1 - 1, upperBoundaryValue);

			final double[] solved =
					ThomasSolver.solve(matrix.lower, matrix.diag, matrix.upper, lineRhs);

			for(int j = 0; j < n1; j++) {
				out[flatten(i, j)] = solved[j];
			}
		}

		return out;
	}

	protected void applyOuterBoundaries(final double time, final double[] u) {
		for(int j = 0; j < n1; j++) {
			u[flatten(0, j)] = getLowerBoundaryValueForFirstDirection(time, j, u[flatten(0, j)]);
			u[flatten(n0 - 1, j)] = getUpperBoundaryValueForFirstDirection(time, j, u[flatten(n0 - 1, j)]);
		}

		for(int i = 0; i < n0; i++) {
			u[flatten(i, 0)] = getLowerBoundaryValueForSecondDirection(time, i, u[flatten(i, 0)]);
			u[flatten(i, n1 - 1)] = getUpperBoundaryValueForSecondDirection(time, i, u[flatten(i, n1 - 1)]);
		}
	}

	protected void applyInternalConstraints(final double time, final double[] u) {
		if(!(product instanceof FiniteDifferenceInternalStateConstraint)) {
			return;
		}

		final FiniteDifferenceInternalStateConstraint constraint =
				(FiniteDifferenceInternalStateConstraint) product;

		for(int j = 0; j < n1; j++) {
			for(int i = 0; i < n0; i++) {
				final int k = flatten(i, j);
				if(constraint.isConstraintActive(time, x0Grid[i], x1Grid[j])) {
					u[k] = constraint.getConstrainedValue(time, x0Grid[i], x1Grid[j]);
				}
			}
		}
	}

	protected void applyExerciseObstacleIfNeeded(
			final double runningTime,
			final double tau,
			final double[] u,
			final DoubleTernaryOperator exerciseValue) {

		final boolean isExerciseAllowed =
				FiniteDifferenceExerciseUtil.isExerciseAllowedAtTimeToMaturity(tau, exercise);

		if(!isExerciseAllowed) {
			return;
		}

		for(int j = 0; j < n1; j++) {
			for(int i = 0; i < n0; i++) {
				if(isInternalConstraintActive(runningTime, x0Grid[i], x1Grid[j])) {
					continue;
				}

				final int k = flatten(i, j);
				final double payoff = exerciseValue.applyAsDouble(runningTime, x0Grid[i], x1Grid[j]);
				u[k] = Math.max(u[k], payoff);
			}
		}
	}

	protected boolean isInternalConstraintActive(final double time, final double x0, final double x1) {
		if(product instanceof FiniteDifferenceInternalStateConstraint) {
			return ((FiniteDifferenceInternalStateConstraint) product).isConstraintActive(time, x0, x1);
		}
		return false;
	}

	protected double getLowerBoundaryValueForFirstDirection(
			final double time,
			final int secondIndex,
			final double fallback) {
		final BoundaryCondition[] conditions =
				model.getBoundaryConditionsAtLowerBoundary(product, time, x0Grid[0], x1Grid[secondIndex]);
		return extractBoundaryValue(conditions[0], fallback);
	}

	protected double getUpperBoundaryValueForFirstDirection(
			final double time,
			final int secondIndex,
			final double fallback) {
		final BoundaryCondition[] conditions =
				model.getBoundaryConditionsAtUpperBoundary(product, time, x0Grid[n0 - 1], x1Grid[secondIndex]);
		return extractBoundaryValue(conditions[0], fallback);
	}

	protected double getLowerBoundaryValueForSecondDirection(
			final double time,
			final int firstIndex,
			final double fallback) {
		final BoundaryCondition[] conditions =
				model.getBoundaryConditionsAtLowerBoundary(product, time, x0Grid[firstIndex], x1Grid[0]);
		return extractBoundaryValue(conditions[1], fallback);
	}

	protected double getUpperBoundaryValueForSecondDirection(
			final double time,
			final int firstIndex,
			final double fallback) {
		final BoundaryCondition[] conditions =
				model.getBoundaryConditionsAtUpperBoundary(product, time, x0Grid[firstIndex], x1Grid[n1 - 1]);
		return extractBoundaryValue(conditions[1], fallback);
	}

	protected double extractBoundaryValue(final BoundaryCondition condition, final double fallback) {
		if(condition != null && condition.isDirichlet()) {
			return condition.getValue();
		}
		return fallback;
	}

	protected void overwriteBoundaryRow(
			final TridiagonalMatrix matrix,
			final double[] rhs,
			final int row,
			final double value) {
		matrix.lower[row] = 0.0;
		matrix.diag[row] = 1.0;
		matrix.upper[row] = 0.0;
		rhs[row] = value;
	}

	protected int flatten(final int i0, final int i1) {
		return i0 + i1 * n0;
	}

	protected double[] add(final double[] a, final double[] b) {
		final double[] out = new double[a.length];
		for(int i = 0; i < a.length; i++) {
			out[i] = a[i] + b[i];
		}
		return out;
	}

	protected double[] subtract(final double[] a, final double[] b) {
		final double[] out = new double[a.length];
		for(int i = 0; i < a.length; i++) {
			out[i] = a[i] - b[i];
		}
		return out;
	}

	protected double[] scale(final double[] a, final double c) {
		final double[] out = new double[a.length];
		for(int i = 0; i < a.length; i++) {
			out[i] = c * a[i];
		}
		return out;
	}

	protected double[] sanitize(final double[] u) {
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