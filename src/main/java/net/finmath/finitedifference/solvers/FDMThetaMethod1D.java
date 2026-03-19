package net.finmath.finitedifference.solvers;

import java.util.function.DoubleUnaryOperator;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.DecompositionSolver;
import org.apache.commons.math3.linear.DiagonalMatrix;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import net.finmath.finitedifference.FiniteDifferenceExerciseUtil;
import net.finmath.finitedifference.assetderivativevaluation.boundaries.FiniteDifferenceBoundaryConditionAdapter;
import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
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

		final FiniteDifferenceMatrixBuilder fdBuilder = new FiniteDifferenceMatrixBuilder(xGrid);
		final RealMatrix T1 = fdBuilder.getFirstDerivativeMatrix();
		final RealMatrix T2 = fdBuilder.getSecondDerivativeMatrix();
		final RealMatrix I = MatrixUtils.createRealIdentityMatrix(nX);

		RealMatrix U = MatrixUtils.createRealMatrix(nX, 1);
		for(int i = 0; i < nX; i++) {
			U.setEntry(i, 0, valueAtMaturity.applyAsDouble(xGrid[i]));
		}

		final RealMatrix z = MatrixUtils.createRealMatrix(nX, timeLength);
		z.setColumnMatrix(0, U);

		for(int m = 0; m < M; m++) {

			final double deltaTau = spaceTimeDiscretization.getTimeDiscretization().getTimeStep(m);

			final double t_m = spaceTimeDiscretization.getTimeDiscretization().getTime(M - m);
			final double t_mp1 = spaceTimeDiscretization.getTimeDiscretization().getTime(M - (m + 1));

			final double tSafe_m = (t_m == 0.0 ? 1e-6 : t_m);
			final double tSafe_mp1 = (t_mp1 == 0.0 ? 1e-6 : t_mp1);

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

			final RealMatrix Mu_m = new DiagonalMatrix(mu_m);
			final RealMatrix Mu_mp1 = new DiagonalMatrix(mu_mp1);
			final RealMatrix A_m = new DiagonalMatrix(a_m);
			final RealMatrix A_mp1 = new DiagonalMatrix(a_mp1);

			final RealMatrix driftTerm_m = Mu_m.scalarMultiply(deltaTau).multiply(T1);
			final RealMatrix driftTerm_mp1 = Mu_mp1.scalarMultiply(deltaTau).multiply(T1);

			final RealMatrix diffTerm_m = A_m.scalarMultiply(0.5 * deltaTau).multiply(T2);
			final RealMatrix diffTerm_mp1 = A_mp1.scalarMultiply(0.5 * deltaTau).multiply(T2);

			final RealMatrix F = I.scalarMultiply(1.0 - r_m * deltaTau).add(driftTerm_m).add(diffTerm_m);
			final RealMatrix G = I.scalarMultiply(1.0 + r_mp1 * deltaTau).subtract(driftTerm_mp1).subtract(diffTerm_mp1);

			RealMatrix H = G.scalarMultiply(theta).add(I.scalarMultiply(1.0 - theta));
			final RealMatrix A = F.scalarMultiply(1.0 - theta).add(I.scalarMultiply(theta));

			RealMatrix rhs = A.multiply(U);

			final double tau_mp1 = spaceTimeDiscretization.getTimeDiscretization().getTime(m + 1);
			final double boundaryTime = spaceTimeDiscretization.getTimeDiscretization().getLastTime() - tau_mp1;

			// Lower boundary
			final BoundaryCondition lowerCondition =
					FiniteDifferenceBoundaryConditionAdapter.getLowerBoundaryConditions(
							(net.finmath.finitedifference.assetderivativevaluation.boundaries.FiniteDifferenceBoundary) model,
							product,
							boundaryTime,
							1,
							xGrid[0])[0];

			if(lowerCondition.isDirichlet()) {
				for(int col = 0; col < nX; col++) {
					H.setEntry(0, col, 0.0);
				}
				H.setEntry(0, 0, 1.0);
				rhs.setEntry(0, 0, lowerCondition.getValue());
			}

			// Upper boundary
			final BoundaryCondition upperCondition =
					FiniteDifferenceBoundaryConditionAdapter.getUpperBoundaryConditions(
							(net.finmath.finitedifference.assetderivativevaluation.boundaries.FiniteDifferenceBoundary) model,
							product,
							boundaryTime,
							1,
							xGrid[nX - 1])[0];

			if(upperCondition.isDirichlet()) {
				for(int col = 0; col < nX; col++) {
					H.setEntry(nX - 1, col, 0.0);
				}
				H.setEntry(nX - 1, nX - 1, 1.0);
				rhs.setEntry(nX - 1, 0, upperCondition.getValue());
			}

			final boolean isExerciseDate =
					FiniteDifferenceExerciseUtil.isExerciseAllowedAtTimeToMaturity(tau_mp1, exercise);

			if(exercise.isAmerican()) {
				final double omega = 1.2;
				final SORDecomposition sor = new SORDecomposition(H);
				final RealMatrix zz = sor.getSol(U, rhs, omega, 500);

				if(isExerciseDate) {
					for(int i = 0; i < nX; i++) {
						if(i == 0 && lowerCondition.isDirichlet()) {
							U.setEntry(i, 0, lowerCondition.getValue());
						}
						else if(i == nX - 1 && upperCondition.isDirichlet()) {
							U.setEntry(i, 0, upperCondition.getValue());
						}
						else {
							U.setEntry(i, 0, Math.max(zz.getEntry(i, 0), valueAtMaturity.applyAsDouble(xGrid[i])));
						}
					}
				}
				else {
					U = zz;
				}
			}
			else {
				final DecompositionSolver solver = new LUDecomposition(H).getSolver();
				U = solver.solve(rhs);

				if(isExerciseDate) {
					for(int i = 0; i < nX; i++) {
						if(i == 0 && lowerCondition.isDirichlet()) {
							U.setEntry(i, 0, lowerCondition.getValue());
						}
						else if(i == nX - 1 && upperCondition.isDirichlet()) {
							U.setEntry(i, 0, upperCondition.getValue());
						}
						else {
							U.setEntry(i, 0, Math.max(U.getEntry(i, 0), valueAtMaturity.applyAsDouble(xGrid[i])));
						}
					}
				}
			}

			z.setColumnMatrix(m + 1, U);
		}

		return z.getData();
	}

	@Override
	public double[] getValue(
			final double evaluationTime,
			final double time,
			final DoubleUnaryOperator valueAtMaturity) {

		final RealMatrix values = new Array2DRowRealMatrix(getValues(time, valueAtMaturity));

		final double tau = time - evaluationTime;
		final int timeIndex = this.spaceTimeDiscretization.getTimeDiscretization().getTimeIndexNearestLessOrEqual(tau);

		return values.getColumn(timeIndex);
	}
}