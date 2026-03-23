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
 * Finite difference solver for two-dimensional Heston-type pricing problems
 * based on an alternating direction implicit (ADI) scheme on the state
 * variables spot and variance.
 * <p>
 * The solver works on a two-dimensional space grid {@code (S, v)} supplied by a
 * {@link SpaceTimeDiscretization}. It evolves the pricing surface backward in
 * time from the terminal payoff to earlier evaluation times using a
 * stabilization-first Douglas-type ADI splitting. The implementation is
 * designed for products driven by an {@link FDMHestonModel} and supports
 * European, Bermudan, and American-style exercise features as well as internal
 * state constraints such as barriers.
 * </p>
 *
 * <p>
 * The infinitesimal generator is split into three parts:
 * </p>
 * <ul>
 *   <li>
 *     {@code A0}: the mixed derivative contribution together with the discount term,
 *     treated explicitly,
 *   </li>
 *   <li>
 *     {@code A1}: the spot-direction drift and diffusion contribution,
 *     treated implicitly by line solves along the spot direction,
 *   </li>
 *   <li>
 *     {@code A2}: the variance-direction drift and diffusion contribution,
 *     treated implicitly by line solves along the variance direction.
 *   </li>
 * </ul>
 *
 * <p>
 * Each full time step is split into two half-sized Douglas steps. This
 * stabilization-first strategy is intended to improve robustness for the
 * two-dimensional Heston PDE, in particular in the presence of mixed derivative
 * terms and non-uniform spatial grids.
 * </p>
 *
 * <p>
 * The internal storage uses a flattened one-dimensional representation of the
 * two-dimensional solution surface. The flattening convention is
 * </p>
 * <pre>
 * k = iS + iV * nS
 * </pre>
 * <p>
 * where {@code iS} denotes the spot-grid index and {@code iV} denotes the
 * variance-grid index. Hence the spot index is the fastest-moving index.
 * </p>
 *
 * <p>
 * Boundary handling is performed through the boundary conditions returned by
 * the model at the lower and upper boundaries. Dirichlet conditions are
 * enforced explicitly. If a non-Dirichlet boundary condition is returned, the
 * current value is retained as fallback.
 * </p>
 *
 * <p>
 * For products implementing {@link FiniteDifferenceInternalStateConstraint},
 * internal hard constraints are enforced directly on the grid values. This is
 * intended for situations where the value is prescribed in parts of the state
 * space, for example due to barrier activation or knock-out logic. Such hard
 * constraints take precedence over the early-exercise obstacle.
 * </p>
 *
 * <p>
 * For products with Bermudan or American exercise rights, the solver applies a
 * post-step projection
 * </p>
 * <pre>
 * u = max(u, payoff)
 * </pre>
 * <p>
 * whenever exercise is allowed at the corresponding running time. This
 * projection is applied only after completion of the full stabilized time step.
 * Internal hard constraints and outer boundary values are then re-applied to
 * preserve consistency.
 * </p>
 *
 * <p>
 * The solver sanitizes intermediate and final solution vectors by replacing
 * non-finite values with zero and clipping extreme values to a fixed range.
 * This provides an additional safeguard against numerical instability during
 * the iteration.
 * </p>
 *
 * <p>
 * The implementation expects a genuinely two-dimensional discretization, that
 * is, one spatial grid for spot and one spatial grid for variance.
 * </p>
 *
 * @author Alessandro Gnoatto
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

	/**
	 * Creates a two-dimensional ADI finite difference solver for a Heston-type model.
	 * <p>
	 * The constructor extracts the spot and variance grids from the supplied
	 * {@link SpaceTimeDiscretization}, stores the product and exercise
	 * specification, and initializes the stencil builder used to construct the
	 * one-dimensional implicit line systems. The ADI weighting parameter
	 * {@code theta} is taken from the discretization and bounded from below by
	 * {@code 0.5}.
	 * </p>
	 *
	 * @param model The Heston finite difference model providing drift, factor
	 * 		loadings, discounting, and boundary conditions.
	 * @param product The finite difference product to be valued.
	 * @param spaceTimeDiscretization The time and two-dimensional space
	 * 		discretization used by the solver.
	 * @param exercise The exercise specification governing whether the product is
	 * 		European, Bermudan, or American.
	 * @throws IllegalArgumentException If the space discretization does not
	 * 		provide both a spot grid and a variance grid.
	 */
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

	/**
	 * Computes the full solution surface for a payoff that depends only on the spot.
	 * <p>
	 * This method is a convenience overload delegating to the two-dimensional
	 * payoff version by interpreting the terminal payoff as independent of the
	 * variance state variable.
	 * </p>
	 *
	 * @param time The maturity or terminal time from which the backward solution
	 * 		is started.
	 * @param valueAtMaturity The terminal payoff as a function of the spot only.
	 * @return A two-dimensional array whose columns contain the flattened pricing
	 * 		surface at successive time levels of the backward evolution.
	 */
	@Override
	public double[][] getValues(final double time, final DoubleUnaryOperator valueAtMaturity) {
		return getValues(time, (s, v) -> valueAtMaturity.applyAsDouble(s));
	}

	/**
	 * Computes the full solution surface for a payoff depending on spot and variance.
	 * <p>
	 * The terminal condition is initialized at the supplied terminal time and the
	 * PDE is then solved backward over the complete time discretization. After
	 * each full time step, internal hard constraints, outer boundaries, and the
	 * early-exercise obstacle are enforced in that order, with hard constraints
	 * and boundaries re-applied after the exercise projection.
	 * </p>
	 *
	 * <p>
	 * The returned matrix is organized column-wise in time. Column {@code 0}
	 * contains the terminal condition and each subsequent column contains the
	 * flattened solution after one backward time step.
	 * </p>
	 *
	 * @param time The maturity or terminal time from which the backward solution
	 * 		is started.
	 * @param valueAtMaturity The terminal payoff as a function of spot and
	 * 		variance.
	 * @return A two-dimensional array representing the full time evolution of the
	 * 		flattened solution surface.
	 */
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

	/**
	 * Returns the flattened solution vector at a specified evaluation time for a
	 * payoff depending only on the spot.
	 * <p>
	 * The method first computes the full solution surface and then extracts the
	 * column corresponding to the largest time-to-maturity grid point that is
	 * less than or equal to {@code time - evaluationTime}.
	 * </p>
	 *
	 * @param evaluationTime The time at which the value is requested.
	 * @param time The terminal time or maturity.
	 * @param valueAtMaturity The terminal payoff as a function of the spot only.
	 * @return The flattened solution vector at the requested evaluation time.
	 */
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

	/**
	 * Returns the flattened solution vector at a specified evaluation time for a
	 * payoff depending on both spot and variance.
	 * <p>
	 * The method first computes the full solution surface and then extracts the
	 * column corresponding to the largest time-to-maturity grid point that is
	 * less than or equal to {@code time - evaluationTime}.
	 * </p>
	 *
	 * @param evaluationTime The time at which the value is requested.
	 * @param time The terminal time or maturity.
	 * @param valueAtMaturity The terminal payoff as a function of spot and variance.
	 * @return The flattened solution vector at the requested evaluation time.
	 */
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
	 * Performs one stabilized PDE time step by splitting it into two Douglas
	 * half-steps.
	 * <p>
	 * The first half-step advances the solution over {@code dt / 2} to an
	 * intermediate state, which is then sanitized. The second half-step advances
	 * from the intermediate state over the remaining half-step to the final state,
	 * which is sanitized again before being returned.
	 * </p>
	 *
	 * @param u The current flattened solution vector.
	 * @param currentTime The running time associated with the start of the step.
	 * @param dt The full PDE time step size.
	 * @return The updated flattened solution vector after one stabilized time step.
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

	/**
	 * Performs one Douglas-type ADI half-step.
	 * <p>
	 * The method first applies the full operator explicitly to form an
	 * intermediate vector. It then performs an implicit spot-direction solve
	 * followed by an implicit variance-direction solve. Outer boundary values are
	 * enforced during the half-step, while internal hard constraints are imposed
	 * only at the end of the half-step. Early exercise is intentionally deferred
	 * to the end of the completed full time step.
	 * </p>
	 *
	 * @param u The current flattened solution vector.
	 * @param currentTime The running time associated with this half-step.
	 * @param dt The half-step size.
	 * @return The updated flattened solution vector after one Douglas half-step.
	 */
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

	/**
	 * Applies the full split operator explicitly.
	 * <p>
	 * This method computes the sum of the explicit contributions from the mixed
	 * term and discount component, the spot-direction operator, and the
	 * variance-direction operator.
	 * </p>
	 *
	 * @param u The flattened solution vector.
	 * @param time The running time at which the operator is evaluated.
	 * @return The explicit action of the full operator on {@code u}.
	 */
	protected double[] applyFullExplicitOperator(final double[] u, final double time) {
		return add(add(applyA0Explicit(u, time), applyA1Explicit(u, time)), applyA2Explicit(u, time));
	}

	/**
	 * Applies the explicit mixed-derivative and discount operator.
	 * <p>
	 * This part of the operator contains the spot-variance mixed derivative and
	 * the discount term. The mixed derivative is approximated by a centered
	 * finite difference on the two-dimensional grid. The discount rate is
	 * extracted from the model's discount curve.
	 * </p>
	 *
	 * @param u The flattened solution vector.
	 * @param time The running time at which the operator is evaluated.
	 * @return The explicit action of the mixed-derivative and discount operator.
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
	 * Applies the explicit spot-direction operator.
	 * <p>
	 * This operator contains the spot-direction drift and diffusion terms. The
	 * first derivative and second derivative in the spot direction are
	 * approximated by centered finite differences on the possibly non-uniform
	 * spot grid.
	 * </p>
	 *
	 * @param u The flattened solution vector.
	 * @param time The running time at which the operator is evaluated.
	 * @return The explicit action of the spot-direction operator.
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
	 * Applies the explicit variance-direction operator.
	 * <p>
	 * This operator contains the variance-direction drift and diffusion terms.
	 * The first derivative and second derivative in the variance direction are
	 * approximated by centered finite differences on the possibly non-uniform
	 * variance grid.
	 * </p>
	 *
	 * @param u The flattened solution vector.
	 * @param time The running time at which the operator is evaluated.
	 * @return The explicit action of the variance-direction operator.
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
	 * Solves the implicit spot-direction systems for all fixed variance lines.
	 * <p>
	 * For each variance index, this method constructs the corresponding
	 * tridiagonal line system, inserts the boundary rows, and solves the system
	 * using the {@link ThomasSolver}. The resulting line values are written back
	 * into the flattened output vector.
	 * </p>
	 *
	 * @param rhs The right-hand side vector for the implicit spot solve.
	 * @param time The running time at which the line systems are assembled.
	 * @param dt The time step size associated with the implicit solve.
	 * @return The flattened solution vector after the spot-direction implicit solve.
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
	 * Solves the implicit variance-direction systems for all fixed spot lines.
	 * <p>
	 * For each spot index, this method constructs the corresponding tridiagonal
	 * line system, inserts the boundary rows, and solves the system using the
	 * {@link ThomasSolver}. The resulting line values are written back into the
	 * flattened output vector.
	 * </p>
	 *
	 * @param rhs The right-hand side vector for the implicit variance solve.
	 * @param time The running time at which the line systems are assembled.
	 * @param dt The time step size associated with the implicit solve.
	 * @return The flattened solution vector after the variance-direction implicit solve.
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

	/**
	 * Applies the outer boundary conditions to the current solution vector.
	 * <p>
	 * The method enforces the lower and upper boundary values in the spot
	 * direction for all variance levels and in the variance direction for all
	 * spot levels.
	 * </p>
	 *
	 * @param time The running time at which the boundary values are evaluated.
	 * @param u The flattened solution vector to be modified in place.
	 */
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

	/**
	 * Applies internal hard constraints to the solution vector, if the product
	 * defines such constraints.
	 * <p>
	 * When the product implements
	 * {@link FiniteDifferenceInternalStateConstraint}, each grid point is checked
	 * for constraint activation. If active, the solution value is overwritten by
	 * the constrained value provided by the product.
	 * </p>
	 *
	 * @param time The running time at which the constraints are evaluated.
	 * @param u The flattened solution vector to be modified in place.
	 */
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

	/**
	 * Applies the early-exercise obstacle when exercise is allowed.
	 * <p>
	 * If exercise is permitted at the given time-to-maturity, the method projects
	 * the solution onto the payoff by replacing each unconstrained grid value by
	 * the maximum of the continuation value and the immediate exercise payoff.
	 * Grid points subject to internal hard constraints are not modified.
	 * </p>
	 *
	 * @param runningTime The running time at which the obstacle is evaluated.
	 * @param tau The corresponding time to maturity.
	 * @param u The flattened solution vector to be modified in place.
	 * @param valueAtMaturity The payoff function defining the exercise value.
	 */
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

	/**
	 * Checks whether an internal hard constraint is active at a given grid point.
	 *
	 * @param time The running time.
	 * @param s The spot value.
	 * @param v The variance value.
	 * @return {@code true} if an internal constraint is active at the given state,
	 * 		{@code false} otherwise.
	 */
	protected boolean isInternalConstraintActive(final double time, final double s, final double v) {
		if(product instanceof FiniteDifferenceInternalStateConstraint) {
			return ((FiniteDifferenceInternalStateConstraint) product).isConstraintActive(time, s, v);
		}
		return false;
	}

	/**
	 * Returns the lower boundary value in the spot direction for a fixed variance level.
	 *
	 * @param time The running time.
	 * @param varianceIndex The variance-grid index.
	 * @param fallback The fallback value used if no Dirichlet condition is available.
	 * @return The lower spot boundary value.
	 */
	private double getLowerBoundaryValueForSpot(final double time, final int varianceIndex, final double fallback) {
		final BoundaryCondition[] conditions =
				model.getBoundaryConditionsAtLowerBoundary(product, time, sGrid[0], vGrid[varianceIndex]);
		return extractBoundaryValue(conditions[0], fallback);
	}

	/**
	 * Returns the upper boundary value in the spot direction for a fixed variance level.
	 *
	 * @param time The running time.
	 * @param varianceIndex The variance-grid index.
	 * @param fallback The fallback value used if no Dirichlet condition is available.
	 * @return The upper spot boundary value.
	 */
	private double getUpperBoundaryValueForSpot(final double time, final int varianceIndex, final double fallback) {
		final BoundaryCondition[] conditions =
				model.getBoundaryConditionsAtUpperBoundary(product, time, sGrid[nS - 1], vGrid[varianceIndex]);
		return extractBoundaryValue(conditions[0], fallback);
	}

	/**
	 * Returns the lower boundary value in the variance direction for a fixed spot level.
	 *
	 * @param time The running time.
	 * @param spotIndex The spot-grid index.
	 * @param fallback The fallback value used if no Dirichlet condition is available.
	 * @return The lower variance boundary value.
	 */
	private double getLowerBoundaryValueForVariance(final double time, final int spotIndex, final double fallback) {
		final BoundaryCondition[] conditions =
				model.getBoundaryConditionsAtLowerBoundary(product, time, sGrid[spotIndex], vGrid[0]);
		return extractBoundaryValue(conditions[1], fallback);
	}

	/**
	 * Returns the upper boundary value in the variance direction for a fixed spot level.
	 *
	 * @param time The running time.
	 * @param spotIndex The spot-grid index.
	 * @param fallback The fallback value used if no Dirichlet condition is available.
	 * @return The upper variance boundary value.
	 */
	private double getUpperBoundaryValueForVariance(final double time, final int spotIndex, final double fallback) {
		final BoundaryCondition[] conditions =
				model.getBoundaryConditionsAtUpperBoundary(product, time, sGrid[spotIndex], vGrid[nV - 1]);
		return extractBoundaryValue(conditions[1], fallback);
	}

	/**
	 * Extracts a boundary value from a boundary condition if it is Dirichlet.
	 *
	 * @param condition The boundary condition.
	 * @param fallback The fallback value used if the condition is {@code null} or
	 * 		not Dirichlet.
	 * @return The extracted Dirichlet value or the fallback value.
	 */
	private double extractBoundaryValue(final BoundaryCondition condition, final double fallback) {
		if(condition != null && condition.isDirichlet()) {
			return condition.getValue();
		}
		return fallback;
	}

	/**
	 * Replaces a matrix row by a Dirichlet boundary row and updates the right-hand side.
	 *
	 * @param m The tridiagonal matrix to be modified.
	 * @param rhs The right-hand side vector to be modified.
	 * @param row The row index to overwrite.
	 * @param value The prescribed boundary value.
	 */
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

	/**
	 * Maps a pair of spot and variance indices to the corresponding flattened index.
	 *
	 * @param iS The spot-grid index.
	 * @param iV The variance-grid index.
	 * @return The flattened one-dimensional index.
	 */
	private int flatten(final int iS, final int iV) {
		return iS + iV * nS;
	}

	/**
	 * Returns the component-wise sum of two vectors.
	 *
	 * @param a The first vector.
	 * @param b The second vector.
	 * @return A new vector containing {@code a[i] + b[i]}.
	 */
	private double[] add(final double[] a, final double[] b) {
		final double[] out = new double[a.length];
		for(int i = 0; i < a.length; i++) {
			out[i] = a[i] + b[i];
		}
		return out;
	}

	/**
	 * Returns the component-wise difference of two vectors.
	 *
	 * @param a The first vector.
	 * @param b The second vector.
	 * @return A new vector containing {@code a[i] - b[i]}.
	 */
	private double[] subtract(final double[] a, final double[] b) {
		final double[] out = new double[a.length];
		for(int i = 0; i < a.length; i++) {
			out[i] = a[i] - b[i];
		}
		return out;
	}

	/**
	 * Returns a scalar multiple of a vector.
	 *
	 * @param a The input vector.
	 * @param c The scaling factor.
	 * @return A new vector containing {@code c * a[i]}.
	 */
	private double[] scale(final double[] a, final double c) {
		final double[] out = new double[a.length];
		for(int i = 0; i < a.length; i++) {
			out[i] = c * a[i];
		}
		return out;
	}

	/**
	 * Sanitizes a solution vector by removing non-finite values and clipping
	 * excessively large magnitudes.
	 * <p>
	 * Non-finite values are replaced by {@code 0.0}. Finite values are clipped to
	 * the interval {@code [-1E12, 1E12]}.
	 * </p>
	 *
	 * @param u The vector to sanitize.
	 * @return A sanitized copy of the input vector.
	 */
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