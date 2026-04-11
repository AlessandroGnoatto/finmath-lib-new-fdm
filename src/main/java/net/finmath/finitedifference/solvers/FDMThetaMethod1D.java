package net.finmath.finitedifference.solvers;

import java.util.function.DoubleUnaryOperator;

import net.finmath.finitedifference.FiniteDifferenceExerciseUtil;
import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.assetderivativevaluation.products.FiniteDifferenceInternalStateConstraint;
import net.finmath.finitedifference.assetderivativevaluation.products.FiniteDifferenceProduct;
import net.finmath.finitedifference.boundaries.BoundaryCondition;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.modelling.Exercise;

/**
 * Theta-method solver for a one-dimensional PDE in <em>state-variable form</em>.
 *
 * <p>
 * This solver assumes the grid variable
 * {@code X} follows an SDE of the form
 * </p>
 *
 * <p>
 * {@code dX_t = mu(t, X_t) dt + sum_k b_k(t, X_t) dW_t^k}
 * </p>
 *
 * <p>
 * and constructs the backward PDE operator using
 * </p>
 *
 * <ul>
 *   <li>Drift term: {@code mu(t, x) * d/dx}</li>
 *   <li>Diffusion term: {@code 0.5 * a(t, x) * d^2/dx^2} where {@code a = sum_k b_k^2}</li>
 *   <li>Discounting term: {@code -r(t) * u}</li>
 * </ul>
 *
 * <p>
 * This makes the solver agnostic to whether {@code X} is {@code S}, {@code log S}, or any other monotone
 * transformation, as long as the model provides consistent coefficients for that chosen state variable.
 * </p>
 *
 * <p>
 * Boundary conditions are enforced via explicit {@link BoundaryCondition} objects.
 * Dirichlet rows are overwritten only if the corresponding boundary condition is of Dirichlet type.
 * If the boundary condition type is NONE, the PDE row is left intact.
 * </p>
 *
 * <p>
 * In addition, products may define internal state constraints through
 * {@link FiniteDifferenceInternalStateConstraint}. Constrained nodes are imposed
 * as internal Dirichlet rows.
 * </p>
 *
 * <p>
 * This implementation is matrix-free and assembles the theta-step directly as a tridiagonal system.
 * </p>
 *
 * @author Alessandro Gnoatto
 * @author Ralph Rudd
 * @author Christian Fries
 * @author Jörg Kienitz
 */
public class FDMThetaMethod1D implements FDMSolver {

	private final FiniteDifferenceEquityModel model;
	private final FiniteDifferenceProduct product;
	private final SpaceTimeDiscretization spaceTimeDiscretization;
	private final Exercise exercise;

	/**
	 * Creates a theta-method finite-difference solver for a one-dimensional backward PDE.
	 *
	 * <p>
	 * The supplied model provides the local drift, variance, discounting, and boundary conditions
	 * for the chosen state variable. The product may additionally define internal state constraints,
	 * and the exercise specification determines whether exercise projection is applied at eligible dates.
	 * </p>
	 *
	 * @param model The finite-difference equity model providing PDE coefficients and boundary conditions.
	 * @param product The product to be valued. May optionally implement
	 *        {@link FiniteDifferenceInternalStateConstraint} to impose internal Dirichlet constraints.
	 * @param spaceTimeDiscretization The joint spatial and temporal discretization, including the theta parameter.
	 * @param exercise The exercise specification controlling whether and when exercise is allowed.
	 */
	public FDMThetaMethod1D(
			final FiniteDifferenceEquityModel model,
			final FiniteDifferenceProduct product,
			final SpaceTimeDiscretization spaceTimeDiscretization,
			final Exercise exercise) {
		this.model = model;
		this.product = product;
		this.spaceTimeDiscretization = spaceTimeDiscretization;
		this.exercise = exercise;
	}

	/**
	 * Solves the backward PDE on the full space-time grid and returns the computed value surface.
	 *
	 * <p>
	 * The method initializes the terminal condition from {@code valueAtMaturity}, then steps backward
	 * in time using the theta scheme.
	 * </p>
	 *
	 * @param time The maturity time of the claim.
	 * @param valueAtMaturity The terminal payoff function as a function of the spatial grid variable.
	 * @return The full space-time value surface on the finite-difference grid.
	 */
	@Override
	public double[][] getValues(final double time, final DoubleUnaryOperator valueAtMaturity) {

		final double[] xGrid = spaceTimeDiscretization.getSpaceGrid(0).getGrid();
		final double[] terminalValues = new double[xGrid.length];

		for(int i = 0; i < xGrid.length; i++) {
			terminalValues[i] = valueAtMaturity.applyAsDouble(xGrid[i]);
		}

		return getValues(time, terminalValues, valueAtMaturity);
	}

	/**
	 * Solves the backward PDE on the full space-time grid using a precomputed
	 * terminal value vector.
	 *
	 * <p>
	 * This overload is intended for products requiring a non-pointwise terminal
	 * initialization, for example cell-averaged digital payoffs.
	 * </p>
	 *
	 * @param time The maturity time of the claim.
	 * @param terminalValues The terminal values on the spatial grid.
	 * @return The full space-time value surface on the finite-difference grid.
	 */
	@Override
	public double[][] getValues(final double time, final double[] terminalValues) {
		return getValues(time, terminalValues, null);
	}

	/**
	 * Shared implementation for terminal-function and terminal-vector initialization.
	 *
	 * @param time The maturity time of the claim.
	 * @param terminalValues The terminal values on the spatial grid.
	 * @param valueAtMaturity The intrinsic payoff function used for exercise decisions.
	 *        This may be {@code null} if early exercise is not needed.
	 * @return The full space-time value surface on the finite-difference grid.
	 */
	private double[][] getValues(
			final double time,
			final double[] terminalValues,
			final DoubleUnaryOperator valueAtMaturity) {

		final double[] xGrid = spaceTimeDiscretization.getSpaceGrid(0).getGrid();
		final int nX = xGrid.length;

		if(terminalValues == null) {
			throw new IllegalArgumentException("terminalValues must not be null.");
		}
		if(terminalValues.length != nX) {
			throw new IllegalArgumentException("terminalValues length does not match spatial grid length.");
		}
		if(!exercise.isEuropean() && valueAtMaturity == null) {
			throw new IllegalArgumentException(
					"Early-exercise products require a pointwise payoff function for obstacle/projection handling.");
		}

		final double theta = spaceTimeDiscretization.getTheta();
		final int timeLength = spaceTimeDiscretization.getTimeDiscretization().getNumberOfTimeSteps() + 1;
		final int numberOfTimeSteps = spaceTimeDiscretization.getTimeDiscretization().getNumberOfTimeSteps();

		double[] u = terminalValues.clone();
		final double[][] z = new double[nX][timeLength];

		for(int i = 0; i < nX; i++) {
			z[i][0] = u[i];
		}

		for(int m = 0; m < numberOfTimeSteps; m++) {

			final double deltaTau = spaceTimeDiscretization.getTimeDiscretization().getTimeStep(m);

			final double t_m = spaceTimeDiscretization.getTimeDiscretization().getTime(numberOfTimeSteps - m);
			final double t_mp1 = spaceTimeDiscretization.getTimeDiscretization().getTime(numberOfTimeSteps - (m + 1));

			final ThetaMethod1DAssembly.ModelCoefficients coefficients_m =
					ThetaMethod1DAssembly.buildModelCoefficients(model, xGrid, t_m);
			final ThetaMethod1DAssembly.ModelCoefficients coefficients_mp1 =
					ThetaMethod1DAssembly.buildModelCoefficients(model, xGrid, t_mp1);

			final TridiagonalMatrix lhs = new TridiagonalMatrix(nX);
			final TridiagonalMatrix rhsOperator = new TridiagonalMatrix(nX);

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

			final double[] rhs = ThetaMethod1DAssembly.apply(rhsOperator, u);

			final double tau_mp1 = spaceTimeDiscretization.getTimeDiscretization().getTime(m + 1);
			final double boundaryTime = spaceTimeDiscretization.getTimeDiscretization().getLastTime() - tau_mp1;

			final BoundaryCondition lowerCondition =
					model.getBoundaryConditionsAtLowerBoundary(product, boundaryTime, xGrid[0])[0];

			if(lowerCondition.isDirichlet()) {
				ThetaMethod1DAssembly.overwriteAsDirichlet(lhs, rhs, 0, lowerCondition.getValue());
			}

			final BoundaryCondition upperCondition =
					model.getBoundaryConditionsAtUpperBoundary(product, boundaryTime, xGrid[nX - 1])[0];

			if(upperCondition.isDirichlet()) {
				ThetaMethod1DAssembly.overwriteAsDirichlet(lhs, rhs, nX - 1, upperCondition.getValue());
			}

			for(int i = 1; i < nX - 1; i++) {
				final double x = xGrid[i];
				if(isInternalConstraintActive(boundaryTime, x)) {
					ThetaMethod1DAssembly.overwriteAsDirichlet(lhs, rhs, i, getInternalConstrainedValue(boundaryTime, x));
				}
			}

			final boolean isExerciseDate =
					FiniteDifferenceExerciseUtil.isExerciseAllowedAtTimeToMaturity(tau_mp1, exercise);

			final double[] nextU;
			if(exercise.isAmerican() && isExerciseDate) {

				final double[] obstacle = buildObstacleVector(
						xGrid,
						boundaryTime,
						valueAtMaturity,
						lowerCondition,
						upperCondition);

				nextU = ProjectedTridiagonalSOR.solve(
						lhs,
						rhs,
						obstacle,
						u,
						1.2,
						500,
						1E-10);

				reimposeInternalConstraints(nextU, xGrid, boundaryTime);
				reimposeBoundaryValues(nextU, lowerCondition, upperCondition);
			}
			else {
				nextU = ThomasSolver.solve(lhs.lower, lhs.diag, lhs.upper, rhs);

				if(isExerciseDate) {
					applyExerciseProjection(nextU, xGrid, boundaryTime, valueAtMaturity, lowerCondition, upperCondition);
				}
				else {
					reimposeInternalConstraints(nextU, xGrid, boundaryTime);
					reimposeBoundaryValues(nextU, lowerCondition, upperCondition);
				}
			}

			u = nextU;
			for(int i = 0; i < nX; i++) {
				z[i][m + 1] = u[i];
			}
		}

		return z;
	}

	/**
	 * Returns the value vector at a specific evaluation time by extracting the appropriate time slice
	 * from the full space-time solution.
	 *
	 * @param evaluationTime The time at which the value is requested.
	 * @param time The maturity time of the claim.
	 * @param valueAtMaturity The terminal payoff function as a function of the spatial grid variable.
	 * @return The value vector across the spatial grid at the requested evaluation time.
	 */
	@Override
	public double[] getValue(
			final double evaluationTime,
			final double time,
			final DoubleUnaryOperator valueAtMaturity) {

		final double[][] values = getValues(time, valueAtMaturity);

		final double tau = time - evaluationTime;
		final int timeIndex = this.spaceTimeDiscretization.getTimeDiscretization().getTimeIndexNearestLessOrEqual(tau);

		final double[] column = new double[values.length];
		for(int i = 0; i < values.length; i++) {
			column[i] = values[i][timeIndex];
		}
		return column;
	}

	/**
	 * Returns the value vector at a specific evaluation time using a precomputed
	 * terminal value vector.
	 *
	 * @param evaluationTime The time at which the value is requested.
	 * @param time The maturity time of the claim.
	 * @param terminalValues The terminal values on the spatial grid.
	 * @return The value vector across the spatial grid at the requested evaluation time.
	 */
	@Override
	public double[] getValue(
			final double evaluationTime,
			final double time,
			final double[] terminalValues) {

		final double[][] values = getValues(time, terminalValues);

		final double tau = time - evaluationTime;
		final int timeIndex = this.spaceTimeDiscretization.getTimeDiscretization().getTimeIndexNearestLessOrEqual(tau);

		final double[] column = new double[values.length];
		for(int i = 0; i < values.length; i++) {
			column[i] = values[i][timeIndex];
		}
		return column;
	}

	/**
	 * Builds the obstacle vector used in the projected solve for American exercise.
	 *
	 * @param xGrid The spatial grid.
	 * @param boundaryTime The current backward time level expressed in model time.
	 * @param valueAtMaturity The intrinsic payoff function used as exercise value.
	 * @param lowerCondition The boundary condition at the lower grid boundary.
	 * @param upperCondition The boundary condition at the upper grid boundary.
	 * @return The obstacle vector for the projected tridiagonal solver.
	 */
	private double[] buildObstacleVector(
			final double[] xGrid,
			final double boundaryTime,
			final DoubleUnaryOperator valueAtMaturity,
			final BoundaryCondition lowerCondition,
			final BoundaryCondition upperCondition) {

		final double[] obstacle = new double[xGrid.length];
		for(int i = 0; i < xGrid.length; i++) {
			if(i == 0 && lowerCondition.isDirichlet()) {
				obstacle[i] = lowerCondition.getValue();
			}
			else if(i == xGrid.length - 1 && upperCondition.isDirichlet()) {
				obstacle[i] = upperCondition.getValue();
			}
			else if(isInternalConstraintActive(boundaryTime, xGrid[i])) {
				obstacle[i] = getInternalConstrainedValue(boundaryTime, xGrid[i]);
			}
			else {
				obstacle[i] = valueAtMaturity.applyAsDouble(xGrid[i]);
			}
		}
		return obstacle;
	}

	/**
	 * Applies pointwise exercise projection to a solution vector at an exercise date.
	 *
	 * @param u The solution vector to be modified in place.
	 * @param xGrid The spatial grid corresponding to {@code u}.
	 * @param boundaryTime The current backward time level expressed in model time.
	 * @param valueAtMaturity The intrinsic payoff function used as exercise value.
	 * @param lowerCondition The boundary condition at the lower grid boundary.
	 * @param upperCondition The boundary condition at the upper grid boundary.
	 */
	private void applyExerciseProjection(
			final double[] u,
			final double[] xGrid,
			final double boundaryTime,
			final DoubleUnaryOperator valueAtMaturity,
			final BoundaryCondition lowerCondition,
			final BoundaryCondition upperCondition) {

		for(int i = 0; i < xGrid.length; i++) {
			if(i == 0 && lowerCondition.isDirichlet()) {
				u[i] = lowerCondition.getValue();
			}
			else if(i == xGrid.length - 1 && upperCondition.isDirichlet()) {
				u[i] = upperCondition.getValue();
			}
			else if(isInternalConstraintActive(boundaryTime, xGrid[i])) {
				u[i] = getInternalConstrainedValue(boundaryTime, xGrid[i]);
			}
			else {
				u[i] = Math.max(u[i], valueAtMaturity.applyAsDouble(xGrid[i]));
			}
		}
	}

	/**
	 * Reapplies active internal state constraints to the interior grid nodes of a solution vector.
	 *
	 * @param u The solution vector to be modified in place.
	 * @param xGrid The spatial grid corresponding to {@code u}.
	 * @param boundaryTime The current backward time level expressed in model time.
	 */
	private void reimposeInternalConstraints(
			final double[] u,
			final double[] xGrid,
			final double boundaryTime) {

		for(int i = 1; i < xGrid.length - 1; i++) {
			if(isInternalConstraintActive(boundaryTime, xGrid[i])) {
				u[i] = getInternalConstrainedValue(boundaryTime, xGrid[i]);
			}
		}
	}

	/**
	 * Reapplies active Dirichlet boundary values to the solution vector.
	 *
	 * @param u The solution vector to be modified in place.
	 * @param lowerCondition The lower boundary condition.
	 * @param upperCondition The upper boundary condition.
	 */
	private void reimposeBoundaryValues(
			final double[] u,
			final BoundaryCondition lowerCondition,
			final BoundaryCondition upperCondition) {

		if(lowerCondition.isDirichlet()) {
			u[0] = lowerCondition.getValue();
		}
		if(upperCondition.isDirichlet()) {
			u[u.length - 1] = upperCondition.getValue();
		}
	}

	/**
	 * Checks whether the product defines an active internal state constraint at the given time and state.
	 *
	 * @param time The model time at which the constraint is queried.
	 * @param x The spatial state variable value.
	 * @return {@code true} if an internal constraint is active at the specified point, {@code false} otherwise.
	 */
	private boolean isInternalConstraintActive(final double time, final double x) {
		if(product instanceof FiniteDifferenceInternalStateConstraint) {
			return ((FiniteDifferenceInternalStateConstraint) product).isConstraintActive(time, x);
		}
		return false;
	}

	/**
	 * Returns the value prescribed by the product's internal state constraint at the given time and state.
	 *
	 * @param time The model time at which the constrained value is queried.
	 * @param x The spatial state variable value.
	 * @return The constrained value to be imposed at the specified point.
	 */
	private double getInternalConstrainedValue(final double time, final double x) {
		return ((FiniteDifferenceInternalStateConstraint) product).getConstrainedValue(time, x);
	}
}