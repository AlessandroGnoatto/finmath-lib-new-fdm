package net.finmath.finitedifference.solvers;

import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

import net.finmath.finitedifference.FiniteDifferenceExerciseUtil;
import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.assetderivativevaluation.products.FiniteDifferenceInternalStateConstraint;
import net.finmath.finitedifference.assetderivativevaluation.products.FiniteDifferenceEquityProduct;
import net.finmath.finitedifference.boundaries.BoundaryCondition;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.modelling.Exercise;

/**
 * Theta-method solver for a one-dimensional PDE in state-variable form.
 *
 * <p>
 * The solver supports:
 * </p>
 * <ul>
 *   <li>pointwise terminal payoff initialization,</li>
 *   <li>direct terminal-vector initialization,</li>
 *   <li>direct terminal-vector initialization with separate pointwise exercise payoff,</li>
 *   <li>direct terminal-vector initialization with a continuous time-dependent obstacle.</li>
 * </ul>
 *
 * <p>
 * The third case is useful for Bermudan and American digitals, where the maturity
 * layer should be cell-averaged, while early-exercise projection should remain pointwise.
 * </p>
 *
 * <p>
 * The fourth case is useful for shout-style problems, where the solution is constrained
 * by a time- and state-dependent continuation floor V &gt;= V*(t,x) at every time step.
 * </p>
 *
 * @author Alessandro Gnoatto
 * @author Ralph Rudd
 * @author Christian Fries
 * @author Jörg Kienitz
 */
public class FDMThetaMethod1D implements FDMSolver {

	private final FiniteDifferenceEquityModel model;
	private final FiniteDifferenceEquityProduct product;
	private final SpaceTimeDiscretization spaceTimeDiscretization;
	private final Exercise exercise;

	/**
	 * Creates a theta-method finite-difference solver for a one-dimensional backward PDE.
	 *
	 * @param model The finite-difference model providing PDE coefficients and boundary conditions.
	 * @param product The product to be valued.
	 * @param spaceTimeDiscretization The joint space-time discretization.
	 * @param exercise The exercise specification.
	 */
	public FDMThetaMethod1D(
			final FiniteDifferenceEquityModel model,
			final FiniteDifferenceEquityProduct product,
			final SpaceTimeDiscretization spaceTimeDiscretization,
			final Exercise exercise) {
		this.model = model;
		this.product = product;
		this.spaceTimeDiscretization = spaceTimeDiscretization;
		this.exercise = exercise;
	}

	@Override
	public double[][] getValues(final double time, final DoubleUnaryOperator valueAtMaturity) {

		final double[] xGrid = spaceTimeDiscretization.getSpaceGrid(0).getGrid();
		final double[] terminalValues = new double[xGrid.length];

		for(int i = 0; i < xGrid.length; i++) {
			terminalValues[i] = valueAtMaturity.applyAsDouble(xGrid[i]);
		}

		return getValuesInternal(time, terminalValues, valueAtMaturity, null);
	}

	@Override
	public double[] getValue(
			final double evaluationTime,
			final double time,
			final DoubleUnaryOperator valueAtMaturity) {

		final double[][] values = getValues(time, valueAtMaturity);
		return extractTimeSlice(values, time, evaluationTime);
	}

	@Override
	public double[][] getValues(final double time, final double[] terminalValues) {
		return getValuesInternal(time, terminalValues, null, null);
	}

	@Override
	public double[] getValue(
			final double evaluationTime,
			final double time,
			final double[] terminalValues) {

		final double[][] values = getValues(time, terminalValues);
		return extractTimeSlice(values, time, evaluationTime);
	}

	@Override
	public double[][] getValues(
			final double time,
			final double[] terminalValues,
			final DoubleUnaryOperator exerciseValue) {
		return getValuesInternal(time, terminalValues, exerciseValue, null);
	}

	@Override
	public double[] getValue(
			final double evaluationTime,
			final double time,
			final double[] terminalValues,
			final DoubleUnaryOperator exerciseValue) {

		final double[][] values = getValues(time, terminalValues, exerciseValue);
		return extractTimeSlice(values, time, evaluationTime);
	}

	public double[][] getValues(
			final double time,
			final double[] terminalValues,
			final DoubleBinaryOperator continuousObstacleValue) {
		return getValuesInternal(time, terminalValues, null, continuousObstacleValue);
	}

	public double[] getValue(
			final double evaluationTime,
			final double time,
			final double[] terminalValues,
			final DoubleBinaryOperator continuousObstacleValue) {

		final double[][] values = getValues(time, terminalValues, continuousObstacleValue);
		return extractTimeSlice(values, time, evaluationTime);
	}

	private double[] extractTimeSlice(
			final double[][] values,
			final double time,
			final double evaluationTime) {

		final double tau = time - evaluationTime;
		final int timeIndex = spaceTimeDiscretization.getTimeDiscretization().getTimeIndexNearestLessOrEqual(tau);

		final double[] column = new double[values.length];
		for(int i = 0; i < values.length; i++) {
			column[i] = values[i][timeIndex];
		}
		return column;
	}

	private double[][] getValuesInternal(
			final double time,
			final double[] terminalValues,
			final DoubleUnaryOperator exerciseValue,
			final DoubleBinaryOperator continuousObstacleValue) {

		final double[] xGrid = spaceTimeDiscretization.getSpaceGrid(0).getGrid();
		final int nX = xGrid.length;

		if(terminalValues == null) {
			throw new IllegalArgumentException("terminalValues must not be null.");
		}
		if(terminalValues.length != nX) {
			throw new IllegalArgumentException("terminalValues length does not match spatial grid length.");
		}
		if(exerciseValue != null && continuousObstacleValue != null) {
			throw new IllegalArgumentException(
					"Provide either a discrete exercise payoff or a continuous obstacle, not both.");
		}
		if((exercise.isBermudan() || exercise.isAmerican())
				&& exerciseValue == null
				&& continuousObstacleValue == null) {
			throw new IllegalArgumentException(
					"Non-European exercise requires a pointwise exercise payoff function.");
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

			if(continuousObstacleValue != null) {
				final double[] obstacle = buildContinuousObstacleVector(
						xGrid,
						boundaryTime,
						continuousObstacleValue,
						lowerCondition,
						upperCondition
				);

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
			else if(exercise.isAmerican() && isExerciseDate) {

				final double[] obstacle = buildObstacleVector(
						xGrid,
						boundaryTime,
						exerciseValue,
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

				if(isExerciseDate && (exercise.isBermudan() || exercise.isAmerican())) {
					applyExerciseProjection(
							nextU,
							xGrid,
							boundaryTime,
							exerciseValue,
							lowerCondition,
							upperCondition);
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

	private double[] buildObstacleVector(
			final double[] xGrid,
			final double boundaryTime,
			final DoubleUnaryOperator exerciseValue,
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
				obstacle[i] = exerciseValue.applyAsDouble(xGrid[i]);
			}
		}
		return obstacle;
	}

	private double[] buildContinuousObstacleVector(
			final double[] xGrid,
			final double boundaryTime,
			final DoubleBinaryOperator continuousObstacleValue,
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
				obstacle[i] = continuousObstacleValue.applyAsDouble(boundaryTime, xGrid[i]);
			}
		}

		return obstacle;
	}

	private void applyExerciseProjection(
			final double[] u,
			final double[] xGrid,
			final double boundaryTime,
			final DoubleUnaryOperator exerciseValue,
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
				u[i] = Math.max(u[i], exerciseValue.applyAsDouble(xGrid[i]));
			}
		}
	}

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

	private boolean isInternalConstraintActive(final double time, final double x) {
		if(product instanceof FiniteDifferenceInternalStateConstraint) {
			return ((FiniteDifferenceInternalStateConstraint) product).isConstraintActive(time, x);
		}
		return false;
	}

	private double getInternalConstrainedValue(final double time, final double x) {
		return ((FiniteDifferenceInternalStateConstraint) product).getConstrainedValue(time, x);
	}
}