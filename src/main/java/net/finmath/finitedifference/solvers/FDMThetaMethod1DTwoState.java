package net.finmath.finitedifference.solvers;

import java.util.function.DoubleUnaryOperator;

import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.assetderivativevaluation.products.BarrierOption;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.modelling.Exercise;
import net.finmath.modelling.products.BarrierType;
import net.finmath.finitedifference.FiniteDifferenceExerciseUtil;
import net.finmath.finitedifference.FiniteDifferenceExerciseUtil;

/**
 * Direct two-state theta-method solver for 1D knock-in barrier options.
 *
 * <p>
 * Regime 0 = not yet activated (barrier not yet hit).
 * Regime 1 = already activated (barrier has been hit).
 * </p>
 *
 * <p>
 * This implementation is matrix-free:
 * </p>
 *
 * <ul>
 *   <li>The active regime is solved on the full grid using a tridiagonal theta step.</li>
 *   <li>The inactive regime is solved only on the continuation-side subgrid.</li>
 *   <li>On the already-hit region, the inactive regime is eliminated by imposing {@code inactive = active}.</li>
 * </ul>
 *
 * <p>
 * Supports European, Bermudan, and American exercise in the active regime.
 * </p>
 * 
 * @author Alessandro Gnoatto
 */
public class FDMThetaMethod1DTwoState implements FDMSolver {

	private static final double EPSILON = 1E-10;

	private final FiniteDifferenceEquityModel model;
	private final BarrierOption product;
	private final SpaceTimeDiscretization spaceTimeDiscretization;
	private final Exercise exercise;
	private final TwoStateActiveBoundaryProvider activeBoundaryProvider;

	/**
	 * Creates a direct two-state theta-method solver for one-dimensional knock-in barrier options.
	 *
	 * <p>
	 * The solver evolves two coupled value functions:
	 * </p>
	 * <ul>
	 *   <li>the <em>inactive</em> regime, representing the contract value before the barrier has been hit,</li>
	 *   <li>the <em>active</em> regime, representing the value after activation, which behaves like the corresponding vanilla claim.</li>
	 * </ul>
	 *
	 * <p>
	 * The active regime is solved on the full spatial grid, while the inactive regime is solved only on the
	 * portion of the grid where the barrier has not yet been triggered. On the already-hit region, the inactive
	 * value is identified with the active one.
	 * </p>
	 *
	 * @param model The finite-difference equity model providing local PDE coefficients and discounting.
	 * @param product The knock-in barrier option to be valued.
	 * @param spaceTimeDiscretization The spatial and temporal discretization, including the theta parameter.
	  * @param exercise The exercise specification. Bermudan and American exercise are applied only to the active regime.
	 * @param activeBoundaryProvider Provider for the boundary values of the already-activated regime.
	 */
	public FDMThetaMethod1DTwoState(
			final FiniteDifferenceEquityModel model,
			final BarrierOption product,
			final SpaceTimeDiscretization spaceTimeDiscretization,
			final Exercise exercise,
			final TwoStateActiveBoundaryProvider activeBoundaryProvider) {
		this.model = model;
		this.product = product;
		this.spaceTimeDiscretization = spaceTimeDiscretization;
		this.exercise = exercise;
		this.activeBoundaryProvider = activeBoundaryProvider;
	}

	/**
	 * Solves the two-state backward PDE on the full space-time grid and returns the inactive-regime value surface.
	 *
	 * <p>
	 * At maturity, the active regime equals the plain payoff, while the inactive regime equals either
	 * the payoff on the already-hit region or the contract rebate on the not-yet-hit region.
	 * The method then steps backward in time:
	 * </p>
	 * <ul>
	 *   <li>solving the activated regime on the full grid,</li>
	 *   <li>solving the non-activated regime on the continuation-side subgrid,</li>
	 *   <li>imposing the coupling condition {@code inactive = active} on the already-hit region.</li>
	 * </ul>
	 *
	 * <p>
	 * The returned surface stores the inactive-regime values only, since these represent the contract
	 * value prior to barrier activation.
	 * </p>
	 *
	 * @param time The maturity time of the product. This parameter is part of the solver interface and is
	 *        used consistently with the supplied terminal payoff.
	 * @param valueAtMaturity The terminal payoff as a function of the state variable.
	 * @return The inactive-regime solution surface indexed as {@code values[spaceIndex][timeIndex]}.
	 * @throws IllegalArgumentException If the exercise style is not European, if no active boundary provider
	 *         is supplied, if the barrier type is unsupported, if the grid is too small, or if the barrier
	 *         does not coincide with a grid point.
	 */
	@Override
	public double[][] getValues(final double time, final DoubleUnaryOperator valueAtMaturity) {

		if(activeBoundaryProvider == null) {
			throw new IllegalArgumentException("Active boundary provider must not be null.");
		}

		if(!exercise.isEuropean() && !exercise.isBermudan() && !exercise.isAmerican()) {
			throw new IllegalArgumentException(
					"FDMThetaMethod1DTwoState currently supports only European, Bermudan, and American exercise.");
		}

		final BarrierType barrierType = product.getBarrierType();
		if(barrierType != BarrierType.DOWN_IN && barrierType != BarrierType.UP_IN) {
			throw new IllegalArgumentException("FDMThetaMethod1DTwoState is only for knock-in barrier options.");
		}

		final double[] xGrid = spaceTimeDiscretization.getSpaceGrid(0).getGrid();
		final int nX = xGrid.length;
		if(nX < 2) {
			throw new IllegalArgumentException("Need at least two grid points.");
		}

		final int barrierIndex = findBarrierIndex(xGrid, product.getBarrierValue());
		final int timeLength = spaceTimeDiscretization.getTimeDiscretization().getNumberOfTimeSteps() + 1;
		final int numberOfTimeSteps = spaceTimeDiscretization.getTimeDiscretization().getNumberOfTimeSteps();

		double[] inactive = new double[nX];
		double[] active = new double[nX];

		for(int i = 0; i < nX; i++) {
			final double x = xGrid[i];
			final double payoff = valueAtMaturity.applyAsDouble(x);

			active[i] = payoff;
			inactive[i] = isAlreadyHitRegion(x, barrierType, product.getBarrierValue()) ? payoff : product.getRebate();
		}

		final double[][] solutionSurface = new double[nX][timeLength];
		for(int i = 0; i < nX; i++) {
			solutionSurface[i][0] = inactive[i];
		}

		for(int m = 0; m < numberOfTimeSteps; m++) {

			final double deltaTau = spaceTimeDiscretization.getTimeDiscretization().getTimeStep(m);

			final double t_m = spaceTimeDiscretization.getTimeDiscretization().getTime(numberOfTimeSteps - m);
			final double t_mp1 = spaceTimeDiscretization.getTimeDiscretization().getTime(numberOfTimeSteps - (m + 1));

			final double tau_mp1 = spaceTimeDiscretization.getTimeDiscretization().getTime(m + 1);
			final double currentTime = spaceTimeDiscretization.getTimeDiscretization().getLastTime() - tau_mp1;

			final double lowerActiveBoundary = activeBoundaryProvider.getLowerBoundaryValue(currentTime, xGrid[0]);
			final double upperActiveBoundary = activeBoundaryProvider.getUpperBoundaryValue(currentTime, xGrid[nX - 1]);

			final boolean isExerciseDate =
					FiniteDifferenceExerciseUtil.isExerciseAllowedAtTimeToMaturity(tau_mp1, exercise);

			final double[] nextActive;
			if(exercise.isAmerican() && isExerciseDate) {
				nextActive = solveVanillaStepAmerican(
						xGrid,
						active,
						t_m,
						t_mp1,
						deltaTau,
						lowerActiveBoundary,
						upperActiveBoundary,
						valueAtMaturity
				);
			}
			else {
				final double[] nextActiveContinuation = solveVanillaStep(
						xGrid,
						active,
						t_m,
						t_mp1,
						deltaTau,
						lowerActiveBoundary,
						upperActiveBoundary
				);

				if(exercise.isBermudan() && isExerciseDate) {
					nextActive = applyBermudanExerciseProjection(
							nextActiveContinuation,
							xGrid,
							valueAtMaturity,
							lowerActiveBoundary,
							upperActiveBoundary
					);
				}
				else {
					nextActive = nextActiveContinuation;
				}
			}

			final double[] nextInactive = new double[nX];

			switch(barrierType) {
			case DOWN_IN:
				fillDownInInactiveStep(
						xGrid,
						barrierIndex,
						inactive,
						active,
						nextActive,
						nextInactive,
						t_m,
						t_mp1,
						deltaTau,
						currentTime);
				break;

			case UP_IN:
				fillUpInInactiveStep(
						xGrid,
						barrierIndex,
						inactive,
						active,
						nextActive,
						nextInactive,
						t_m,
						t_mp1,
						deltaTau,
						currentTime);
				break;

			default:
				throw new IllegalArgumentException("Unsupported barrier type: " + barrierType);
			}

			active = nextActive;
			inactive = nextInactive;

			for(int i = 0; i < nX; i++) {
				solutionSurface[i][m + 1] = inactive[i];
			}
		}

		return solutionSurface;
	}

	/**
	 * Returns the inactive-regime value vector at the requested evaluation time.
	 *
	 * <p>
	 * This method computes the full inactive-regime space-time surface using
	 * {@link #getValues(double, DoubleUnaryOperator)} and then extracts the column corresponding to the
	 * nearest discretized time-to-maturity less than or equal to {@code time - evaluationTime}.
	 * </p>
	 *
	 * @param evaluationTime The time at which the value vector is requested.
	 * @param time The maturity time of the claim.
	 * @param valueAtMaturity The terminal payoff as a function of the state variable.
	 * @return The inactive-regime value vector across the spatial grid at the specified evaluation time.
	 */
	@Override
	public double[] getValue(
			final double evaluationTime,
			final double time,
			final DoubleUnaryOperator valueAtMaturity) {

		final double[][] values = getValues(time, valueAtMaturity);
		final double tau = time - evaluationTime;
		final int timeIndex = spaceTimeDiscretization.getTimeDiscretization().getTimeIndexNearestLessOrEqual(tau);

		final double[] column = new double[values.length];
		for(int i = 0; i < values.length; i++) {
			column[i] = values[i][timeIndex];
		}
		return column;
	}

	/**
	 * Advances the inactive regime by one time step for a down-in barrier option.
	 *
	 * <p>
	 * For a down-in option, all states at or below the barrier belong to the already-hit region and therefore
	 * satisfy {@code inactive = active}. Only the states strictly above the barrier remain in the non-activated
	 * continuation region and require a PDE solve on the corresponding subgrid.
	 * </p>
	 *
	 * <p>
	 * The lower boundary of the continuation subproblem is coupled to the active regime at the barrier,
	 * while the upper boundary corresponds to the discounted value of remaining unactivated until maturity.
	 * </p>
	 *
	 * @param xGrid The full spatial grid.
	 * @param barrierIndex The index of the barrier node on the spatial grid.
	 * @param inactivePrevious The inactive-regime solution at the previous time level.
	 * @param activePrevious The active-regime solution at the previous time level.
	 * @param activeNext The active-regime solution at the new time level.
	 * @param inactiveNext Output array receiving the inactive-regime solution at the new time level.
	 * @param t_m The current backward time level used for the right-hand-side operator.
	 * @param t_mp1 The next backward time level used for the left-hand-side operator.
	 * @param deltaTau The time step size in time-to-maturity coordinates.
	 * @param currentTime The corresponding forward model time.
	 */
	private void fillDownInInactiveStep(
			final double[] xGrid,
			final int barrierIndex,
			final double[] inactivePrevious,
			final double[] activePrevious,
			final double[] activeNext,
			final double[] inactiveNext,
			final double t_m,
			final double t_mp1,
			final double deltaTau,
			final double currentTime) {

		for(int i = 0; i <= barrierIndex; i++) {
			inactiveNext[i] = activeNext[i];
		}

		if(barrierIndex == xGrid.length - 1) {
			return;
		}

		final double[] subGrid = sliceGrid(xGrid, barrierIndex, xGrid.length - 1);
		final double[] previousSub = new double[subGrid.length];

		for(int j = 0; j < subGrid.length; j++) {
			previousSub[j] = inactivePrevious[barrierIndex + j];
		}

		previousSub[0] = activePrevious[barrierIndex];

		final double discountedNoHitValue = getDiscountedNoHitValue(currentTime);

		final double[] nextSub = solveVanillaStep(
				subGrid,
				previousSub,
				t_m,
				t_mp1,
				deltaTau,
				activeNext[barrierIndex],
				discountedNoHitValue);

		for(int j = 0; j < nextSub.length; j++) {
			inactiveNext[barrierIndex + j] = nextSub[j];
		}

		for(int i = 0; i <= barrierIndex; i++) {
			inactiveNext[i] = activeNext[i];
		}
	}

	/**
	 * Advances the inactive regime by one time step for an up-in barrier option.
	 *
	 * <p>
	 * For an up-in option, all states at or above the barrier belong to the already-hit region and therefore
	 * satisfy {@code inactive = active}. Only the states strictly below the barrier remain in the non-activated
	 * continuation region and require a PDE solve on the corresponding subgrid.
	 * </p>
	 *
	 * <p>
	 * The upper boundary of the continuation subproblem is coupled to the active regime at the barrier,
	 * while the lower boundary corresponds to the discounted value of remaining unactivated until maturity.
	 * </p>
	 *
	 * @param xGrid The full spatial grid.
	 * @param barrierIndex The index of the barrier node on the spatial grid.
	 * @param inactivePrevious The inactive-regime solution at the previous time level.
	 * @param activePrevious The active-regime solution at the previous time level.
	 * @param activeNext The active-regime solution at the new time level.
	 * @param inactiveNext Output array receiving the inactive-regime solution at the new time level.
	 * @param t_m The current backward time level used for the right-hand-side operator.
	 * @param t_mp1 The next backward time level used for the left-hand-side operator.
	 * @param deltaTau The time step size in time-to-maturity coordinates.
	 * @param currentTime The corresponding forward model time.
	 */
	private void fillUpInInactiveStep(
			final double[] xGrid,
			final int barrierIndex,
			final double[] inactivePrevious,
			final double[] activePrevious,
			final double[] activeNext,
			final double[] inactiveNext,
			final double t_m,
			final double t_mp1,
			final double deltaTau,
			final double currentTime) {

		for(int i = barrierIndex; i < xGrid.length; i++) {
			inactiveNext[i] = activeNext[i];
		}

		if(barrierIndex == 0) {
			return;
		}

		final double[] subGrid = sliceGrid(xGrid, 0, barrierIndex);
		final double[] previousSub = new double[subGrid.length];

		for(int j = 0; j < subGrid.length; j++) {
			previousSub[j] = inactivePrevious[j];
		}

		previousSub[subGrid.length - 1] = activePrevious[barrierIndex];

		final double discountedNoHitValue = getDiscountedNoHitValue(currentTime);

		final double[] nextSub = solveVanillaStep(
				subGrid,
				previousSub,
				t_m,
				t_mp1,
				deltaTau,
				discountedNoHitValue,
				activeNext[barrierIndex]);

		for(int j = 0; j < nextSub.length; j++) {
			inactiveNext[j] = nextSub[j];
		}

		for(int i = barrierIndex; i < xGrid.length; i++) {
			inactiveNext[i] = activeNext[i];
		}
	}

	/**
	 * Solves one theta-method step for a vanilla one-dimensional tridiagonal PDE problem on a given grid.
	 *
	 * <p>
	 * The method assembles the left- and right-hand-side theta operators, applies Dirichlet boundary
	 * conditions at the two grid endpoints, and solves the resulting tridiagonal linear system.
	 * </p>
	 *
	 * <p>
	 * Degenerate grids with one or two nodes are handled explicitly by returning the boundary values directly.
	 * </p>
	 *
	 * @param xGrid The spatial grid for the current subproblem.
	 * @param previousValues The solution vector at the previous time level.
	 * @param t_m The current backward time level used for the right-hand-side operator.
	 * @param t_mp1 The next backward time level used for the left-hand-side operator.
	 * @param deltaTau The time step size in time-to-maturity coordinates.
	 * @param lowerBoundaryValue The Dirichlet value at the lower grid boundary.
	 * @param upperBoundaryValue The Dirichlet value at the upper grid boundary.
	 * @return The solution vector at the next time level.
	 * @throws IllegalArgumentException If the grid and value vector sizes do not match.
	 */
	private double[] solveVanillaStep(
			final double[] xGrid,
			final double[] previousValues,
			final double t_m,
			final double t_mp1,
			final double deltaTau,
			final double lowerBoundaryValue,
			final double upperBoundaryValue) {

		final int n = xGrid.length;

		if(n != previousValues.length) {
			throw new IllegalArgumentException("Grid and solution vector size mismatch.");
		}

		if(n == 1) {
			return new double[] { lowerBoundaryValue };
		}

		if(n == 2) {
			return new double[] { lowerBoundaryValue, upperBoundaryValue };
		}

		final double theta = spaceTimeDiscretization.getTheta();

		final ThetaMethod1DAssembly.ModelCoefficients coefficients_m =
				ThetaMethod1DAssembly.buildModelCoefficients(model, xGrid, t_m);
		final ThetaMethod1DAssembly.ModelCoefficients coefficients_mp1 =
				ThetaMethod1DAssembly.buildModelCoefficients(model, xGrid, t_mp1);

		final TridiagonalMatrix lhs = new TridiagonalMatrix(n);
		final TridiagonalMatrix rhsOperator = new TridiagonalMatrix(n);

		ThetaMethod1DAssembly.buildThetaLeftHandSide(
				lhs,
				xGrid,
				coefficients_mp1.getDrift(),
				coefficients_mp1.getVariance(),
				coefficients_mp1.getShortRate(),
				deltaTau,
				theta);

		ThetaMethod1DAssembly.buildThetaRightHandSide(
				rhsOperator,
				xGrid,
				coefficients_m.getDrift(),
				coefficients_m.getVariance(),
				coefficients_m.getShortRate(),
				deltaTau,
				theta);

		final double[] rhs = ThetaMethod1DAssembly.apply(rhsOperator, previousValues);

		ThetaMethod1DAssembly.overwriteAsDirichlet(lhs, rhs, 0, lowerBoundaryValue);
		ThetaMethod1DAssembly.overwriteAsDirichlet(lhs, rhs, n - 1, upperBoundaryValue);

		final double[] next = ThomasSolver.solve(lhs.lower, lhs.diag, lhs.upper, rhs);
		next[0] = lowerBoundaryValue;
		next[n - 1] = upperBoundaryValue;

		return next;
	}

	/**
	 * Returns the discounted value of the no-hit payoff used at the outer boundary of the inactive regime.
	 *
	 * <p>
	 * This value corresponds to the rebate paid at maturity if the barrier has never been triggered.
	 * It is discounted from maturity back to the current time using the model's risk-free curve.
	 * </p>
	 *
	 * <p>
	 * If the rebate is zero, the method returns zero immediately. If the current time is at or beyond
	 * maturity, the rebate itself is returned.
	 * </p>
	 *
	 * @param currentTime The current model time at which the boundary value is required.
	 * @return The discounted no-hit rebate value.
	 */
	private double getDiscountedNoHitValue(final double currentTime) {

		if(product.getRebate() == 0.0) {
			return 0.0;
		}

		final double t = Math.max(currentTime, EPSILON);
		final double maturity = product.getMaturity();

		if(t >= maturity) {
			return product.getRebate();
		}

		final double discountFactorAtCurrentTime = model.getRiskFreeCurve().getDiscountFactor(t);
		final double discountFactorAtMaturity = model.getRiskFreeCurve().getDiscountFactor(maturity);

		return product.getRebate() * discountFactorAtMaturity / discountFactorAtCurrentTime;
	}

	/**
	 * Finds the index of the barrier on the spatial grid.
	 *
	 * <p>
	 * This direct two-state implementation requires the barrier level to coincide exactly with a grid node,
	 * up to a small numerical tolerance.
	 * </p>
	 *
	 * @param grid The spatial grid.
	 * @param barrier The barrier level.
	 * @return The index of the grid node matching the barrier.
	 * @throws IllegalArgumentException If the barrier does not coincide with any grid node.
	 */
	private int findBarrierIndex(final double[] grid, final double barrier) {
		for(int i = 0; i < grid.length; i++) {
			if(Math.abs(grid[i] - barrier) < 1E-12) {
				return i;
			}
		}

		throw new IllegalArgumentException(
				"Barrier must coincide with a 1D grid node for direct two-state knock-in pricing.");
	}

	/**
	 * Returns a contiguous slice of the spatial grid.
	 *
	 * @param grid The full spatial grid.
	 * @param startInclusive The first index to include.
	 * @param endInclusive The last index to include.
	 * @return A new array containing {@code grid[startInclusive]}, ..., {@code grid[endInclusive]}.
	 */
	private double[] sliceGrid(final double[] grid, final int startInclusive, final int endInclusive) {
		final double[] result = new double[endInclusive - startInclusive + 1];
		for(int i = 0; i < result.length; i++) {
			result[i] = grid[startInclusive + i];
		}
		return result;
	}

	/**
	 * Determines whether a state lies in the already-hit region for the specified barrier type.
	 *
	 * <p>
	 * For a down-in barrier, the already-hit region is {@code x <= barrier}. For an up-in barrier,
	 * it is {@code x >= barrier}.
	 * </p>
	 *
	 * @param x The current state variable value.
	 * @param barrierType The barrier type.
	 * @param barrier The barrier level.
	 * @return {@code true} if the barrier is already considered hit at {@code x}, {@code false} otherwise.
	 * @throws IllegalArgumentException If the barrier type is unsupported.
	 */
	private boolean isAlreadyHitRegion(
			final double x,
			final BarrierType barrierType,
			final double barrier) {

		switch(barrierType) {
		case DOWN_IN:
			return x <= barrier;
		case UP_IN:
			return x >= barrier;
		default:
			throw new IllegalArgumentException("Unsupported barrier type: " + barrierType);
		}
	}

	private double[] solveVanillaStepAmerican(
			final double[] xGrid,
			final double[] previousValues,
			final double t_m,
			final double t_mp1,
			final double deltaTau,
			final double lowerBoundaryValue,
			final double upperBoundaryValue,
			final DoubleUnaryOperator exerciseValue) {

		final int n = xGrid.length;

		if(n != previousValues.length) {
			throw new IllegalArgumentException("Grid and solution vector size mismatch.");
		}

		if(n == 1) {
			return new double[] { lowerBoundaryValue };
		}

		if(n == 2) {
			return new double[] { lowerBoundaryValue, upperBoundaryValue };
		}

		final double theta = spaceTimeDiscretization.getTheta();

		final ThetaMethod1DAssembly.ModelCoefficients coefficients_m =
				ThetaMethod1DAssembly.buildModelCoefficients(model, xGrid, t_m);
		final ThetaMethod1DAssembly.ModelCoefficients coefficients_mp1 =
				ThetaMethod1DAssembly.buildModelCoefficients(model, xGrid, t_mp1);

		final TridiagonalMatrix lhs = new TridiagonalMatrix(n);
		final TridiagonalMatrix rhsOperator = new TridiagonalMatrix(n);

		ThetaMethod1DAssembly.buildThetaLeftHandSide(
				lhs,
				xGrid,
				coefficients_mp1.getDrift(),
				coefficients_mp1.getVariance(),
				coefficients_mp1.getShortRate(),
				deltaTau,
				theta);

		ThetaMethod1DAssembly.buildThetaRightHandSide(
				rhsOperator,
				xGrid,
				coefficients_m.getDrift(),
				coefficients_m.getVariance(),
				coefficients_m.getShortRate(),
				deltaTau,
				theta);

		final double[] rhs = ThetaMethod1DAssembly.apply(rhsOperator, previousValues);

		ThetaMethod1DAssembly.overwriteAsDirichlet(lhs, rhs, 0, lowerBoundaryValue);
		ThetaMethod1DAssembly.overwriteAsDirichlet(lhs, rhs, n - 1, upperBoundaryValue);

		final double[] obstacle = buildActiveObstacleVector(
				xGrid,
				exerciseValue,
				lowerBoundaryValue,
				upperBoundaryValue
		);

		final double[] next = ProjectedTridiagonalSOR.solve(
				lhs,
				rhs,
				obstacle,
				previousValues,
				1.2,
				500,
				1E-10
		);

		next[0] = lowerBoundaryValue;
		next[n - 1] = upperBoundaryValue;

		return next;
	}

	private double[] applyBermudanExerciseProjection(
			final double[] continuationValues,
			final double[] xGrid,
			final DoubleUnaryOperator exerciseValue,
			final double lowerBoundaryValue,
			final double upperBoundaryValue) {

		final double[] exercisedValues = continuationValues.clone();

		for(int i = 0; i < exercisedValues.length; i++) {
			if(i == 0) {
				exercisedValues[i] = lowerBoundaryValue;
			}
			else if(i == exercisedValues.length - 1) {
				exercisedValues[i] = upperBoundaryValue;
			}
			else {
				exercisedValues[i] = Math.max(exercisedValues[i], exerciseValue.applyAsDouble(xGrid[i]));
			}
		}

		return exercisedValues;
	}

	private double[] buildActiveObstacleVector(
			final double[] xGrid,
			final DoubleUnaryOperator exerciseValue,
			final double lowerBoundaryValue,
			final double upperBoundaryValue) {

		final double[] obstacle = new double[xGrid.length];

		for(int i = 0; i < xGrid.length; i++) {
			if(i == 0) {
				obstacle[i] = lowerBoundaryValue;
			}
			else if(i == xGrid.length - 1) {
				obstacle[i] = upperBoundaryValue;
			}
			else {
				obstacle[i] = exerciseValue.applyAsDouble(xGrid[i]);
			}
		}

		return obstacle;
	}
}