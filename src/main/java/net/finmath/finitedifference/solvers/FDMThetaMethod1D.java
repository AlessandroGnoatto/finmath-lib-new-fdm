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

	@Override
	public double[][] getValues(final double time, final DoubleUnaryOperator valueAtMaturity) {

		final double[] xGrid = spaceTimeDiscretization.getSpaceGrid(0).getGrid();
		final int nX = xGrid.length;

		final double theta = spaceTimeDiscretization.getTheta();
		final int timeLength = spaceTimeDiscretization.getTimeDiscretization().getNumberOfTimeSteps() + 1;
		final int M = spaceTimeDiscretization.getTimeDiscretization().getNumberOfTimeSteps();

		double[] u = new double[nX];
		final double[][] z = new double[nX][timeLength];

		for(int i = 0; i < nX; i++) {
			u[i] = valueAtMaturity.applyAsDouble(xGrid[i]);
			z[i][0] = u[i];
		}

		for(int m = 0; m < M; m++) {

			final double deltaTau = spaceTimeDiscretization.getTimeDiscretization().getTimeStep(m);

			final double t_m = spaceTimeDiscretization.getTimeDiscretization().getTime(M - m);
			final double t_mp1 = spaceTimeDiscretization.getTimeDiscretization().getTime(M - (m + 1));

			final double tSafe_m = (t_m == 0.0 ? 1E-6 : t_m);
			final double tSafe_mp1 = (t_mp1 == 0.0 ? 1E-6 : t_mp1);

			final double r_m = -Math.log(model.getRiskFreeCurve().getDiscountFactor(tSafe_m)) / tSafe_m;
			final double r_mp1 = -Math.log(model.getRiskFreeCurve().getDiscountFactor(tSafe_mp1)) / tSafe_mp1;

			final double[] mu_m = new double[nX];
			final double[] mu_mp1 = new double[nX];
			final double[] a_m = new double[nX];
			final double[] a_mp1 = new double[nX];

			for(int i = 0; i < nX; i++) {
				final double x = xGrid[i];

				mu_m[i] = model.getDrift(t_m, x)[0];
				mu_mp1[i] = model.getDrift(t_mp1, x)[0];

				final double[][] b_m = model.getFactorLoading(t_m, x);
				final double[][] b_mp1 = model.getFactorLoading(t_mp1, x);

				double am = 0.0;
				for(int f = 0; f < b_m[0].length; f++) {
					final double b = b_m[0][f];
					am += b * b;
				}

				double ap = 0.0;
				for(int f = 0; f < b_mp1[0].length; f++) {
					final double b = b_mp1[0][f];
					ap += b * b;
				}

				a_m[i] = am;
				a_mp1[i] = ap;
			}

			final TridiagonalMatrix lhs = new TridiagonalMatrix(nX);
			final TridiagonalMatrix rhsOperator = new TridiagonalMatrix(nX);

			buildThetaLeftHandSide(lhs, xGrid, mu_mp1, a_mp1, r_mp1, deltaTau, theta);
			buildThetaRightHandSide(rhsOperator, xGrid, mu_m, a_m, r_m, deltaTau, theta);

			final double[] rhs = apply(rhsOperator, u);

			final double tau_mp1 = spaceTimeDiscretization.getTimeDiscretization().getTime(m + 1);
			final double boundaryTime = spaceTimeDiscretization.getTimeDiscretization().getLastTime() - tau_mp1;

			final BoundaryCondition lowerCondition =
					model.getBoundaryConditionsAtLowerBoundary(product, boundaryTime, xGrid[0])[0];

			if(lowerCondition.isDirichlet()) {
				overwriteAsDirichlet(lhs, rhs, 0, lowerCondition.getValue());
			}

			final BoundaryCondition upperCondition =
					model.getBoundaryConditionsAtUpperBoundary(product, boundaryTime, xGrid[nX - 1])[0];

			if(upperCondition.isDirichlet()) {
				overwriteAsDirichlet(lhs, rhs, nX - 1, upperCondition.getValue());
			}

			for(int i = 1; i < nX - 1; i++) {
				final double x = xGrid[i];
				if(isInternalConstraintActive(boundaryTime, x)) {
					overwriteAsDirichlet(lhs, rhs, i, getInternalConstrainedValue(boundaryTime, x));
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

	private void buildThetaLeftHandSide(
			final TridiagonalMatrix lhs,
			final double[] xGrid,
			final double[] mu,
			final double[] a,
			final double r,
			final double deltaTau,
			final double theta) {

		final double alpha = theta * deltaTau;
		for(int i = 0; i < xGrid.length; i++) {
			final RowCoefficients spatial = spatialOperatorRow(i, xGrid, mu[i], a[i], r);
			lhs.lower[i] = -alpha * spatial.lower;
			lhs.diag[i] = 1.0 - alpha * spatial.diag;
			lhs.upper[i] = -alpha * spatial.upper;
		}
	}

	private void buildThetaRightHandSide(
			final TridiagonalMatrix rhsOperator,
			final double[] xGrid,
			final double[] mu,
			final double[] a,
			final double r,
			final double deltaTau,
			final double theta) {

		final double alpha = (1.0 - theta) * deltaTau;
		for(int i = 0; i < xGrid.length; i++) {
			final RowCoefficients spatial = spatialOperatorRow(i, xGrid, mu[i], a[i], r);
			rhsOperator.lower[i] = alpha * spatial.lower;
			rhsOperator.diag[i] = 1.0 + alpha * spatial.diag;
			rhsOperator.upper[i] = alpha * spatial.upper;
		}
	}

	private RowCoefficients spatialOperatorRow(
			final int i,
			final double[] x,
			final double mu,
			final double variance,
			final double r) {

		final int n = x.length;
		final double halfVariance = 0.5 * variance;

		double t1Lower = 0.0;
		double t1Diag = 0.0;
		double t1Upper = 0.0;

		double t2Lower = 0.0;
		double t2Diag = 0.0;
		double t2Upper = 0.0;

		if(i == 0) {
			final double h1 = x[1] - x[0];
			final double h2 = x[2] - x[1];

			t1Diag = -1.0 / h1;
			t1Upper = 1.0 / h1;

			t2Diag = -2.0 / (h1 * h2);
			t2Upper = 2.0 / (h1 * (h1 + h2));
		}
		else if(i == n - 1) {
			final double h0 = x[i] - x[i - 1];
			final double h3 = x[i - 1] - x[i - 2];

			t1Lower = -1.0 / h0;
			t1Diag = 1.0 / h0;

			t2Lower = 2.0 / (h0 * (h0 + h3));
			t2Diag = -2.0 / (h3 * h0);
		}
		else {
			final double h0 = x[i] - x[i - 1];
			final double h1 = x[i + 1] - x[i];

			t1Lower = -h1 / (h0 * (h1 + h0));
			t1Diag = (h1 - h0) / (h1 * h0);
			t1Upper = h0 / (h1 * (h0 + h1));

			t2Lower = 2.0 / (h0 * (h0 + h1));
			t2Diag = -2.0 / (h0 * h1);
			t2Upper = 2.0 / (h1 * (h0 + h1));
		}

		return new RowCoefficients(
				mu * t1Lower + halfVariance * t2Lower,
				mu * t1Diag + halfVariance * t2Diag - r,
				mu * t1Upper + halfVariance * t2Upper);
	}

	private double[] apply(final TridiagonalMatrix matrix, final double[] vector) {
		final int n = vector.length;
		final double[] result = new double[n];

		for(int i = 0; i < n; i++) {
			double value = matrix.diag[i] * vector[i];
			if(i > 0) {
				value += matrix.lower[i] * vector[i - 1];
			}
			if(i < n - 1) {
				value += matrix.upper[i] * vector[i + 1];
			}
			result[i] = value;
		}

		return result;
	}

	private void overwriteAsDirichlet(
			final TridiagonalMatrix lhs,
			final double[] rhs,
			final int row,
			final double value) {

		lhs.lower[row] = 0.0;
		lhs.diag[row] = 1.0;
		lhs.upper[row] = 0.0;
		rhs[row] = value;
	}

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

	private static final class RowCoefficients {
		private final double lower;
		private final double diag;
		private final double upper;

		private RowCoefficients(final double lower, final double diag, final double upper) {
			this.lower = lower;
			this.diag = diag;
			this.upper = upper;
		}
	}
}