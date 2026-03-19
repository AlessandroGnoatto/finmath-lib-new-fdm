package net.finmath.finitedifference.solvers;

import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.DecompositionSolver;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.OpenMapRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

import net.finmath.finitedifference.FiniteDifferenceExerciseUtil;
import net.finmath.finitedifference.assetderivativevaluation.boundaries.FiniteDifferenceBoundaryConditionAdapter;
import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.assetderivativevaluation.products.FiniteDifferenceProduct;
import net.finmath.finitedifference.boundaries.BoundaryCondition;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.modelling.Exercise;

/**
 * Theta-method solver for two-dimensional PDEs in <em>state-variable form</em>.
 *
 * <p>
 * In contrast to the original {@link FDMThetaMethod2D} (which contained S-specific scaling like {@code S * d/dS} and
 * {@code S^2 * d^2/dS^2}), this solver assumes the two state variables {@code (X0, X1)} follow a generic SDE
 * </p>
 *
 * <p>
 * {@code dX_i(t) = mu_i(t, X0, X1) dt + sum_k b_{i,k}(t, X0, X1) dW_k(t)},  i=0,1,
 * </p>
 *
 * <p>
 * and builds the backward PDE operator using
 * </p>
 *
 * <ul>
 *   <li>Drift: {@code sum_i mu_i * d/dx_i}</li>
 *   <li>Diffusion: {@code 0.5 * sum_{i,j} a_{i,j} * d^2/(dx_i dx_j)} with {@code a = b b^T}</li>
 *   <li>Discounting: {@code -r(t) * u}</li>
 * </ul>
 *
 * <p>
 * Boundary conditions are enforced via explicit {@link BoundaryCondition} objects.
 * Dirichlet rows are overwritten only if the corresponding boundary condition is of Dirichlet type.
 * If the boundary condition type is NONE, the PDE row is left intact.
 * </p>
 *
 * @author Enrico De Vecchi
 * @author Alessandro Gnoatto
 */
public class FDMThetaMethod2D implements FDMSolver {

	private final FiniteDifferenceEquityModel model;
	private final FiniteDifferenceProduct product;
	private final SpaceTimeDiscretization spaceTimeDiscretization;
	private final Exercise exercise;

	public FDMThetaMethod2D(
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
		return getValues(time, (x0, x1) -> valueAtMaturity.applyAsDouble(x0));
	}

	public double[][] getValues(final double time, final DoubleBinaryOperator valueAtMaturity) {

		final Grid x0GridObj = spaceTimeDiscretization.getSpaceGrid(0);
		final Grid x1GridObj = spaceTimeDiscretization.getSpaceGrid(1);
		if (x0GridObj == null || x1GridObj == null) {
			throw new IllegalArgumentException(
					"SpaceTimeDiscretization must provide two space grids (dimension 0 and dimension 1).");
		}

		final double[] x0Grid = x0GridObj.getGrid();
		final double[] x1Grid = x1GridObj.getGrid();

		final int n0 = x0Grid.length;
		final int n1 = x1Grid.length;
		final int n = n0 * n1;

		final double theta = spaceTimeDiscretization.getTheta();

		final int timeLength = spaceTimeDiscretization.getTimeDiscretization().getNumberOfTimeSteps() + 1;
		final int M = spaceTimeDiscretization.getTimeDiscretization().getNumberOfTimeSteps();

		final FiniteDifferenceMatrixBuilder b0 = new FiniteDifferenceMatrixBuilder(x0Grid);
		final RealMatrix T1_0 = b0.getFirstDerivativeMatrix();
		final RealMatrix T2_0 = b0.getSecondDerivativeMatrix();

		final FiniteDifferenceMatrixBuilder b1 = new FiniteDifferenceMatrixBuilder(x1Grid);
		final RealMatrix T1_1 = b1.getFirstDerivativeMatrix();
		final RealMatrix T2_1 = b1.getSecondDerivativeMatrix();

		final RealMatrix D0 = buildBlockDiagonal(T1_0, n1);
		final RealMatrix D00 = buildBlockDiagonal(T2_0, n1);

		final RealMatrix D1 = buildKronWithIdentityLeft(T1_1, n0);
		final RealMatrix D11 = buildKronWithIdentityLeft(T2_1, n0);

		final RealMatrix D01 = buildKron(T1_1, T1_0);

		final boolean[] isBoundary = new boolean[n];
		for (int j = 0; j < n1; j++) {
			for (int i = 0; i < n0; i++) {
				final int k = i + j * n0;
				isBoundary[k] = (i == 0 || i == n0 - 1 || j == 0 || j == n1 - 1);
			}
		}

		RealMatrix U = MatrixUtils.createRealMatrix(n, 1);
		for (int j = 0; j < n1; j++) {
			for (int i = 0; i < n0; i++) {
				final int k = i + j * n0;
				U.setEntry(k, 0, valueAtMaturity.applyAsDouble(x0Grid[i], x1Grid[j]));
			}
		}

		final RealMatrix z = MatrixUtils.createRealMatrix(n, timeLength);
		z.setColumnMatrix(0, U);

		for (int m = 0; m < M; m++) {

			final double deltaTau = spaceTimeDiscretization.getTimeDiscretization().getTimeStep(m);

			final double t_m = spaceTimeDiscretization.getTimeDiscretization().getTime(M - m);
			final double t_mp1 = spaceTimeDiscretization.getTimeDiscretization().getTime(M - (m + 1));

			final double tSafe_m = (t_m == 0.0 ? 1e-6 : t_m);
			final double tSafe_mp1 = (t_mp1 == 0.0 ? 1e-6 : t_mp1);

			final double r_m = -Math.log(model.getRiskFreeCurve().getDiscountFactor(tSafe_m)) / tSafe_m;
			final double r_mp1 = -Math.log(model.getRiskFreeCurve().getDiscountFactor(tSafe_mp1)) / tSafe_mp1;

			final double[] mu0_m = new double[n];
			final double[] mu0_mp1 = new double[n];
			final double[] mu1_m = new double[n];
			final double[] mu1_mp1 = new double[n];

			final double[] a00_m = new double[n];
			final double[] a00_mp1 = new double[n];
			final double[] a11_m = new double[n];
			final double[] a11_mp1 = new double[n];
			final double[] a01_m = new double[n];
			final double[] a01_mp1 = new double[n];

			for (int j = 0; j < n1; j++) {
				for (int i = 0; i < n0; i++) {
					final int k = i + j * n0;
					final double x0 = x0Grid[i];
					final double x1 = x1Grid[j];

					final double[] drift_m = model.getDrift(t_m, x0, x1);
					final double[] drift_mp1 = model.getDrift(t_mp1, x0, x1);

					mu0_m[k] = drift_m.length > 0 ? drift_m[0] : 0.0;
					mu1_m[k] = drift_m.length > 1 ? drift_m[1] : 0.0;

					mu0_mp1[k] = drift_mp1.length > 0 ? drift_mp1[0] : 0.0;
					mu1_mp1[k] = drift_mp1.length > 1 ? drift_mp1[1] : 0.0;

					final double[][] b_m = model.getFactorLoading(t_m, x0, x1);
					final double[][] b_mp1 = model.getFactorLoading(t_mp1, x0, x1);

					double a00v_m = 0.0;
					double a11v_m = 0.0;
					double a01v_m = 0.0;

					final int nFactors_m = b_m[0].length;
					for (int f = 0; f < nFactors_m; f++) {
						final double b00 = b_m[0][f];
						final double b10 = b_m.length > 1 ? b_m[1][f] : 0.0;
						a00v_m += b00 * b00;
						a11v_m += b10 * b10;
						a01v_m += b00 * b10;
					}

					double a00v_p = 0.0;
					double a11v_p = 0.0;
					double a01v_p = 0.0;

					final int nFactors_p = b_mp1[0].length;
					for (int f = 0; f < nFactors_p; f++) {
						final double b00 = b_mp1[0][f];
						final double b10 = b_mp1.length > 1 ? b_mp1[1][f] : 0.0;
						a00v_p += b00 * b00;
						a11v_p += b10 * b10;
						a01v_p += b00 * b10;
					}

					a00_m[k] = a00v_m;
					a11_m[k] = a11v_m;
					a01_m[k] = a01v_m;

					a00_mp1[k] = a00v_p;
					a11_mp1[k] = a11v_p;
					a01_mp1[k] = a01v_p;
				}
			}

			final RealMatrix Mu0_m = MatrixUtils.createRealDiagonalMatrix(mu0_m);
			final RealMatrix Mu0_mp1 = MatrixUtils.createRealDiagonalMatrix(mu0_mp1);
			final RealMatrix Mu1_m = MatrixUtils.createRealDiagonalMatrix(mu1_m);
			final RealMatrix Mu1_mp1 = MatrixUtils.createRealDiagonalMatrix(mu1_mp1);

			final RealMatrix A00_m = MatrixUtils.createRealDiagonalMatrix(a00_m);
			final RealMatrix A00_mp1 = MatrixUtils.createRealDiagonalMatrix(a00_mp1);
			final RealMatrix A11_m = MatrixUtils.createRealDiagonalMatrix(a11_m);
			final RealMatrix A11_mp1 = MatrixUtils.createRealDiagonalMatrix(a11_mp1);
			final RealMatrix A01_m = MatrixUtils.createRealDiagonalMatrix(a01_m);
			final RealMatrix A01_mp1 = MatrixUtils.createRealDiagonalMatrix(a01_mp1);

			final RealMatrix I = MatrixUtils.createRealIdentityMatrix(n);

			final RealMatrix driftTerm_m =
					Mu0_m.scalarMultiply(deltaTau).multiply(D0)
					.add(Mu1_m.scalarMultiply(deltaTau).multiply(D1));

			final RealMatrix driftTerm_mp1 =
					Mu0_mp1.scalarMultiply(deltaTau).multiply(D0)
					.add(Mu1_mp1.scalarMultiply(deltaTau).multiply(D1));

			final RealMatrix diffTerm_m =
					A00_m.multiply(D00.scalarMultiply(0.5 * deltaTau))
					.add(A11_m.multiply(D11.scalarMultiply(0.5 * deltaTau)))
					.add(A01_m.multiply(D01.scalarMultiply(deltaTau)));

			final RealMatrix diffTerm_mp1 =
					A00_mp1.multiply(D00.scalarMultiply(0.5 * deltaTau))
					.add(A11_mp1.multiply(D11.scalarMultiply(0.5 * deltaTau)))
					.add(A01_mp1.multiply(D01.scalarMultiply(deltaTau)));

			final RealMatrix F = I.scalarMultiply(1.0 - r_m * deltaTau).add(driftTerm_m).add(diffTerm_m);
			final RealMatrix G = I.scalarMultiply(1.0 + r_mp1 * deltaTau).subtract(driftTerm_mp1).subtract(diffTerm_mp1);

			RealMatrix H = G.scalarMultiply(theta).add(I.scalarMultiply(1.0 - theta));
			final RealMatrix A = F.scalarMultiply(1.0 - theta).add(I.scalarMultiply(theta));

			RealMatrix rhs = A.multiply(U);

			final double tau_mp1 = spaceTimeDiscretization.getTimeDiscretization().getTime(m + 1);
			final double boundaryTime =
					spaceTimeDiscretization.getTimeDiscretization().getLastTime() - tau_mp1;

			for (int j = 0; j < n1; j++) {
				for (int i = 0; i < n0; i++) {
					final int k = i + j * n0;
					if (!isBoundary[k]) {
						continue;
					}

					final double x0 = x0Grid[i];
					final double x1 = x1Grid[j];

					final BoundaryCondition[] lowerConditions =
							FiniteDifferenceBoundaryConditionAdapter.getLowerBoundaryConditions(
									(net.finmath.finitedifference.assetderivativevaluation.boundaries.FiniteDifferenceBoundary) model,
									product,
									boundaryTime,
									2,
									x0,
									x1);
					final BoundaryCondition[] upperConditions =
							FiniteDifferenceBoundaryConditionAdapter.getUpperBoundaryConditions(
									(net.finmath.finitedifference.assetderivativevaluation.boundaries.FiniteDifferenceBoundary) model,
									product,
									boundaryTime,
									2,
									x0,
									x1);

					final BoundaryCondition x0Lower = lowerConditions[0];
					final BoundaryCondition x0Upper = upperConditions[0];
					final BoundaryCondition x1Lower = lowerConditions[1];
					final BoundaryCondition x1Upper = upperConditions[1];

					BoundaryCondition chosenCondition = null;
					if (i == 0) {
						chosenCondition = x0Lower;
					}
					else if (i == n0 - 1) {
						chosenCondition = x0Upper;
					}
					else if (j == 0) {
						chosenCondition = x1Lower;
					}
					else if (j == n1 - 1) {
						chosenCondition = x1Upper;
					}

					if (chosenCondition != null && chosenCondition.isDirichlet()) {
						for (int col = 0; col < n; col++) {
							H.setEntry(k, col, 0.0);
						}
						H.setEntry(k, k, 1.0);
						rhs.setEntry(k, 0, chosenCondition.getValue());
					}
				}
			}

			final boolean isExerciseDate =
					FiniteDifferenceExerciseUtil.isExerciseAllowedAtTimeToMaturity(tau_mp1, exercise);

			if (exercise.isAmerican()) {
				final double omega = 1.2;
				final SORDecomposition sor = new SORDecomposition(H);
				final RealMatrix zz = sor.getSol(U, rhs, omega, 500);

				if(isExerciseDate) {
					for (int j = 0; j < n1; j++) {
						for (int i = 0; i < n0; i++) {
							final int k = i + j * n0;
							final double payoff = valueAtMaturity.applyAsDouble(x0Grid[i], x1Grid[j]);
							U.setEntry(k, 0, Math.max(zz.getEntry(k, 0), payoff));
						}
					}
				}
				else {
					U = zz;
				}
				z.setColumnMatrix(m + 1, U);
			}
			else {
				final DecompositionSolver solver = new LUDecomposition(H).getSolver();
				U = solver.solve(rhs);

				if(isExerciseDate) {
					for (int j = 0; j < n1; j++) {
						for (int i = 0; i < n0; i++) {
							final int k = i + j * n0;
							final double payoff = valueAtMaturity.applyAsDouble(x0Grid[i], x1Grid[j]);
							U.setEntry(k, 0, Math.max(U.getEntry(k, 0), payoff));
						}
					}
				}
				z.setColumnMatrix(m + 1, U);
			}
		}

		return z.getData();
	}

	@Override
	public double[] getValue(final double evaluationTime, final double time, final DoubleUnaryOperator valueAtMaturity) {
		final RealMatrix values = new Array2DRowRealMatrix(getValues(time, valueAtMaturity));
		final double tau = time - evaluationTime;
		final int timeIndex = this.spaceTimeDiscretization.getTimeDiscretization().getTimeIndexNearestLessOrEqual(tau);
		return values.getColumn(timeIndex);
	}

	private static RealMatrix buildBlockDiagonal(final RealMatrix block, final int numBlocks) {
		final int n = block.getRowDimension();
		final int N = n * numBlocks;
		final OpenMapRealMatrix out = new OpenMapRealMatrix(N, N);

		for (int b = 0; b < numBlocks; b++) {
			final int row0 = b * n;
			final int col0 = b * n;
			for (int i = 0; i < n; i++) {
				for (int j = Math.max(0, i - 2); j <= Math.min(n - 1, i + 2); j++) {
					final double v = block.getEntry(i, j);
					if (v != 0.0) {
						out.setEntry(row0 + i, col0 + j, v);
					}
				}
			}
		}
		return out;
	}

	private static RealMatrix buildKron(final RealMatrix A, final RealMatrix B) {
		final int aR = A.getRowDimension();
		final int aC = A.getColumnDimension();
		final int bR = B.getRowDimension();
		final int bC = B.getColumnDimension();

		final OpenMapRealMatrix out = new OpenMapRealMatrix(aR * bR, aC * bC);

		for (int i = 0; i < aR; i++) {
			for (int j = Math.max(0, i - 2); j <= Math.min(aC - 1, i + 2); j++) {
				final double a = A.getEntry(i, j);
				if (a == 0.0) {
					continue;
				}

				final int rowBase = i * bR;
				final int colBase = j * bC;

				for (int p = 0; p < bR; p++) {
					for (int q = Math.max(0, p - 2); q <= Math.min(bC - 1, p + 2); q++) {
						final double b = B.getEntry(p, q);
						if (b == 0.0) {
							continue;
						}
						out.setEntry(rowBase + p, colBase + q, a * b);
					}
				}
			}
		}
		return out;
	}

	private static RealMatrix buildKronWithIdentityLeft(final RealMatrix A, final int nIdentity) {
		final int aR = A.getRowDimension();
		final int aC = A.getColumnDimension();
		final int N = aR * nIdentity;

		final OpenMapRealMatrix out = new OpenMapRealMatrix(N, aC * nIdentity);

		for (int i = 0; i < aR; i++) {
			for (int j = Math.max(0, i - 2); j <= Math.min(aC - 1, i + 2); j++) {
				final double a = A.getEntry(i, j);
				if (a == 0.0) {
					continue;
				}

				for (int k = 0; k < nIdentity; k++) {
					out.setEntry(i * nIdentity + k, j * nIdentity + k, a);
				}
			}
		}
		return out;
	}
}