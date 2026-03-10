package net.finmath.finitedifference.solvers;

import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.DecompositionSolver;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.OpenMapRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.assetderivativevaluation.products.ExerciseType;
import net.finmath.finitedifference.assetderivativevaluation.products.FiniteDifferenceProduct;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;

/**
 * Constructs and solves matrix systems for the theta method applied to three-dimensional PDEs using finite differences.
 *
 * <p>This solver follows the structure of {@link FDMThetaMethod2D} as closely as possible: it uses
 * {@link FiniteDifferenceEquityModel#getDrift(double, double...)} and
 * {@link FiniteDifferenceEquityModel#getFactorLoading(double, double...)} to obtain the PDE coefficients.</p>
 *
 * <p>It supports generic stochastic-volatility / lifted models with state variables (x0, x1, x2) discretized on
 * three grids including boundary points. The time discretization is assumed to be the time-to-maturity discretization
 * used throughout this project.</p>
 *
 * <p>Boundary conditions are enforced as Dirichlet conditions by overwriting the corresponding rows in the linear
 * system. A boundary row is overwritten only if the boundary provides a finite value for that dimension. If the
 * boundary returns {@link Double#NaN} for a dimension, the row is left intact so that the PDE operator / stencil
 * handles that boundary.</p>
 *
 * <p>The solver returns the full time history as a flattened matrix of dimension (n0 * n1 * n2) x (nT + 1).</p>
 *
 * <p>Flattening convention: index {@code k = i0 + i1*n0 + i2*n0*n1} where {@code i0} is the fastest index.</p>
 *
 * @author Alessandro Gnoatto
 */
public class FDMThetaMethod3D implements FDMSolver {

	/**
	 * Functional interface for terminal payoffs depending on three state variables.
	 */
	@FunctionalInterface
	public interface DoubleTernaryOperator {
		double applyAsDouble(double x0, double x1, double x2);
	}

	private final FiniteDifferenceEquityModel model;
	private final FiniteDifferenceProduct product;
	private final SpaceTimeDiscretization spaceTimeDiscretization;
	private final ExerciseType exercise;

	/**
	 * Creates a 3D theta method solver.
	 *
	 * @param model Finite difference equity model providing drift, factor loadings, and boundary conditions.
	 * @param product Product used for boundary value queries.
	 * @param spaceTimeDiscretization Space-time discretization (three space grids + time discretization).
	 * @param exercise Exercise type (European/American).
	 */
	public FDMThetaMethod3D(
			final FiniteDifferenceEquityModel model,
			final FiniteDifferenceProduct product,
			final SpaceTimeDiscretization spaceTimeDiscretization,
			final ExerciseType exercise) {
		this.model = model;
		this.product = product;
		this.spaceTimeDiscretization = spaceTimeDiscretization;
		this.exercise = exercise;
	}

	@Override
	public double[][] getValues(final double time, final DoubleUnaryOperator valueAtMaturity) {
		return getValues(time, (x0, x1, x2) -> valueAtMaturity.applyAsDouble(x0));
	}

	public double[][] getValues(final double time, final DoubleBinaryOperator valueAtMaturity) {
		return getValues(time, (x0, x1, x2) -> valueAtMaturity.applyAsDouble(x0, x1));
	}

	public double[][] getValues(final double time, final DoubleTernaryOperator valueAtMaturity) {

		final Grid grid0Obj = spaceTimeDiscretization.getSpaceGrid(0);
		final Grid grid1Obj = spaceTimeDiscretization.getSpaceGrid(1);
		final Grid grid2Obj = spaceTimeDiscretization.getSpaceGrid(2);

		if (grid0Obj == null || grid1Obj == null || grid2Obj == null) {
			throw new IllegalArgumentException(
					"SpaceTimeDiscretization must provide three space grids (dimension 0, 1, and 2).");
		}

		final double[] x0Grid = grid0Obj.getGrid();
		final double[] x1Grid = grid1Obj.getGrid();
		final double[] x2Grid = grid2Obj.getGrid();

		final int n0 = x0Grid.length;
		final int n1 = x1Grid.length;
		final int n2 = x2Grid.length;
		final int n = n0 * n1 * n2;

		final double theta = spaceTimeDiscretization.getTheta();

		final int timeLength = spaceTimeDiscretization.getTimeDiscretization().getNumberOfTimeSteps() + 1;
		final int M = spaceTimeDiscretization.getTimeDiscretization().getNumberOfTimeSteps();

		// 1D derivative matrices on FULL grids (including boundary points).
		final FiniteDifferenceMatrixBuilder b0 = new FiniteDifferenceMatrixBuilder(x0Grid);
		final RealMatrix T1_0 = b0.getFirstDerivativeMatrix();
		final RealMatrix T2_0 = b0.getSecondDerivativeMatrix();

		final FiniteDifferenceMatrixBuilder b1 = new FiniteDifferenceMatrixBuilder(x1Grid);
		final RealMatrix T1_1 = b1.getFirstDerivativeMatrix();
		final RealMatrix T2_1 = b1.getSecondDerivativeMatrix();

		final FiniteDifferenceMatrixBuilder b2 = new FiniteDifferenceMatrixBuilder(x2Grid);
		final RealMatrix T1_2 = b2.getFirstDerivativeMatrix();
		final RealMatrix T2_2 = b2.getSecondDerivativeMatrix();

		/*
		 * Build 3D differential operators on flattened state (x0 fastest, then x1, then x2).
		 */
		final RealMatrix Dx0  = buildBlockDiagonal(T1_0, n1 * n2);
		final RealMatrix Dxx0 = buildBlockDiagonal(T2_0, n1 * n2);

		final RealMatrix Dx1  = buildKronWithIdentityLeft(T1_1, n0 * n2);
		final RealMatrix Dxx1 = buildKronWithIdentityLeft(T2_1, n0 * n2);

		final RealMatrix Dx2  = buildKronWithIdentityLeft(T1_2, n0 * n1);
		final RealMatrix Dxx2 = buildKronWithIdentityLeft(T2_2, n0 * n1);

		// Cross derivative operators.
		final RealMatrix Dx01 = buildKronWithIdentityRight(buildKron(T1_1, T1_0), n2); // I2 ⊗ (T1_1 ⊗ T1_0)
		final RealMatrix Dx02 = buildKron(T1_2, MatrixUtils.createRealIdentityMatrix(n1 * n0)); // T1_2 ⊗ I_{n1*n0}
		final RealMatrix Dx12 = buildKronWithIdentityRight(buildKron(T1_2, T1_1), n0); // (T1_2 ⊗ T1_1) ⊗ I0

		// Diagonal matrices for x0 and x0^2 embedded as block diagonal.
		final RealMatrix X0 = buildBlockDiagonal(MatrixUtils.createRealDiagonalMatrix(x0Grid), n1 * n2);
		final double[] x0Sq = new double[n0];
		for (int i = 0; i < n0; i++) {
			x0Sq[i] = x0Grid[i] * x0Grid[i];
		}
		final RealMatrix X0Sq = buildBlockDiagonal(MatrixUtils.createRealDiagonalMatrix(x0Sq), n1 * n2);

		// Identify boundary nodes.
		final boolean[] isBoundary = new boolean[n];
		for (int k2 = 0; k2 < n2; k2++) {
			for (int k1 = 0; k1 < n1; k1++) {
				for (int k0 = 0; k0 < n0; k0++) {
					final int idx = k0 + k1 * n0 + k2 * n0 * n1;
					isBoundary[idx] = (k0 == 0 || k0 == n0 - 1 || k1 == 0 || k1 == n1 - 1 || k2 == 0 || k2 == n2 - 1);
				}
			}
		}

		// Initial condition at maturity (tau=0 <-> t=T).
		RealMatrix U = MatrixUtils.createRealMatrix(n, 1);
		for (int k2 = 0; k2 < n2; k2++) {
			for (int k1 = 0; k1 < n1; k1++) {
				for (int k0 = 0; k0 < n0; k0++) {
					final int idx = k0 + k1 * n0 + k2 * n0 * n1;
					U.setEntry(idx, 0, valueAtMaturity.applyAsDouble(x0Grid[k0], x1Grid[k1], x2Grid[k2]));
				}
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

			final double rF_m = model.getRiskFreeCurve().getDiscountFactor(tSafe_m);
			final double r_m = -Math.log(rF_m) / tSafe_m;
			final double rF_mp1 = model.getRiskFreeCurve().getDiscountFactor(tSafe_mp1);
			final double r_mp1 = -Math.log(rF_mp1) / tSafe_mp1;

			// Percentage drift for x0 evaluated at first grid point (same convention as 1D/2D solvers).
			final double mu0_m = model.getDrift(t_m, x0Grid[0], x1Grid[0], x2Grid[0])[0];
			final double mu0_mp1 = model.getDrift(t_mp1, x0Grid[0], x1Grid[0], x2Grid[0])[0];

			// Drift vectors for x1 and x2.
			final double[] mu1_m_vec = new double[n];
			final double[] mu1_mp1_vec = new double[n];
			final double[] mu2_m_vec = new double[n];
			final double[] mu2_mp1_vec = new double[n];

			// Variances.
			final double[] var00_m = new double[n];
			final double[] var00_mp1 = new double[n];
			final double[] var11_m = new double[n];
			final double[] var11_mp1 = new double[n];
			final double[] var22_m = new double[n];
			final double[] var22_mp1 = new double[n];

			// Covariances.
			final double[] cov01_m = new double[n];
			final double[] cov01_mp1 = new double[n];
			final double[] cov02_m = new double[n];
			final double[] cov02_mp1 = new double[n];
			final double[] cov12_m = new double[n];
			final double[] cov12_mp1 = new double[n];

			for (int k2 = 0; k2 < n2; k2++) {
				for (int k1 = 0; k1 < n1; k1++) {
					for (int k0 = 0; k0 < n0; k0++) {
						final int idx = k0 + k1 * n0 + k2 * n0 * n1;

						final double x0 = x0Grid[k0];
						final double x1 = x1Grid[k1];
						final double x2 = x2Grid[k2];

						final double[] drift_m = model.getDrift(t_m, x0, x1, x2);
						final double[] drift_mp1 = model.getDrift(t_mp1, x0, x1, x2);

						mu1_m_vec[idx] = drift_m.length > 1 ? drift_m[1] : 0.0;
						mu1_mp1_vec[idx] = drift_mp1.length > 1 ? drift_mp1[1] : 0.0;
						mu2_m_vec[idx] = drift_m.length > 2 ? drift_m[2] : 0.0;
						mu2_mp1_vec[idx] = drift_mp1.length > 2 ? drift_mp1[2] : 0.0;

						final double[][] b_m = model.getFactorLoading(t_m, x0, x1, x2);
						final double[][] b_mp1 = model.getFactorLoading(t_mp1, x0, x1, x2);

						double v00_m = 0.0, v11_m = 0.0, v22_m = 0.0;
						double c01_m = 0.0, c02_m = 0.0, c12_m = 0.0;

						double v00_p = 0.0, v11_p = 0.0, v22_p = 0.0;
						double c01_p = 0.0, c02_p = 0.0, c12_p = 0.0;

						final int nFactors_m = b_m[0].length;
						for (int f = 0; f < nFactors_m; f++) {
							final double b0f = b_m[0][f];
							final double b1f = b_m.length > 1 ? b_m[1][f] : 0.0;
							final double b2f = b_m.length > 2 ? b_m[2][f] : 0.0;

							v00_m += b0f * b0f;
							v11_m += b1f * b1f;
							v22_m += b2f * b2f;

							c01_m += (x0 * b0f) * b1f;
							c02_m += (x0 * b0f) * b2f;
							c12_m += b1f * b2f;
						}

						final int nFactors_p = b_mp1[0].length;
						for (int f = 0; f < nFactors_p; f++) {
							final double b0f = b_mp1[0][f];
							final double b1f = b_mp1.length > 1 ? b_mp1[1][f] : 0.0;
							final double b2f = b_mp1.length > 2 ? b_mp1[2][f] : 0.0;

							v00_p += b0f * b0f;
							v11_p += b1f * b1f;
							v22_p += b2f * b2f;

							c01_p += (x0 * b0f) * b1f;
							c02_p += (x0 * b0f) * b2f;
							c12_p += b1f * b2f;
						}

						var00_m[idx] = v00_m;
						var00_mp1[idx] = v00_p;

						var11_m[idx] = v11_m;
						var11_mp1[idx] = v11_p;

						var22_m[idx] = v22_m;
						var22_mp1[idx] = v22_p;

						cov01_m[idx] = c01_m;
						cov01_mp1[idx] = c01_p;

						cov02_m[idx] = c02_m;
						cov02_mp1[idx] = c02_p;

						cov12_m[idx] = c12_m;
						cov12_mp1[idx] = c12_p;
					}
				}
			}

			final RealMatrix Mu1_m = MatrixUtils.createRealDiagonalMatrix(mu1_m_vec);
			final RealMatrix Mu1_mp1 = MatrixUtils.createRealDiagonalMatrix(mu1_mp1_vec);
			final RealMatrix Mu2_m = MatrixUtils.createRealDiagonalMatrix(mu2_m_vec);
			final RealMatrix Mu2_mp1 = MatrixUtils.createRealDiagonalMatrix(mu2_mp1_vec);

			final RealMatrix Var00_m = MatrixUtils.createRealDiagonalMatrix(var00_m);
			final RealMatrix Var00_mp1 = MatrixUtils.createRealDiagonalMatrix(var00_mp1);
			final RealMatrix Var11_m = MatrixUtils.createRealDiagonalMatrix(var11_m);
			final RealMatrix Var11_mp1 = MatrixUtils.createRealDiagonalMatrix(var11_mp1);
			final RealMatrix Var22_m = MatrixUtils.createRealDiagonalMatrix(var22_m);
			final RealMatrix Var22_mp1 = MatrixUtils.createRealDiagonalMatrix(var22_mp1);

			final RealMatrix Cov01_m = MatrixUtils.createRealDiagonalMatrix(cov01_m);
			final RealMatrix Cov01_mp1 = MatrixUtils.createRealDiagonalMatrix(cov01_mp1);
			final RealMatrix Cov02_m = MatrixUtils.createRealDiagonalMatrix(cov02_m);
			final RealMatrix Cov02_mp1 = MatrixUtils.createRealDiagonalMatrix(cov02_mp1);
			final RealMatrix Cov12_m = MatrixUtils.createRealDiagonalMatrix(cov12_m);
			final RealMatrix Cov12_mp1 = MatrixUtils.createRealDiagonalMatrix(cov12_mp1);

			final RealMatrix Id = MatrixUtils.createRealIdentityMatrix(n);

			final RealMatrix driftTerm_m =
					X0.scalarMultiply(mu0_m * deltaTau).multiply(Dx0)
					.add(Mu1_m.scalarMultiply(deltaTau).multiply(Dx1))
					.add(Mu2_m.scalarMultiply(deltaTau).multiply(Dx2));

			final RealMatrix driftTerm_mp1 =
					X0.scalarMultiply(mu0_mp1 * deltaTau).multiply(Dx0)
					.add(Mu1_mp1.scalarMultiply(deltaTau).multiply(Dx1))
					.add(Mu2_mp1.scalarMultiply(deltaTau).multiply(Dx2));

			final RealMatrix diffTerm_m =
					Var00_m.multiply(X0Sq.scalarMultiply(0.5 * deltaTau).multiply(Dxx0))
					.add(Var11_m.multiply(Dxx1.scalarMultiply(0.5 * deltaTau)))
					.add(Var22_m.multiply(Dxx2.scalarMultiply(0.5 * deltaTau)))
					.add(Cov01_m.multiply(Dx01.scalarMultiply(deltaTau)))
					.add(Cov02_m.multiply(Dx02.scalarMultiply(deltaTau)))
					.add(Cov12_m.multiply(Dx12.scalarMultiply(deltaTau)));

			final RealMatrix diffTerm_mp1 =
					Var00_mp1.multiply(X0Sq.scalarMultiply(0.5 * deltaTau).multiply(Dxx0))
					.add(Var11_mp1.multiply(Dxx1.scalarMultiply(0.5 * deltaTau)))
					.add(Var22_mp1.multiply(Dxx2.scalarMultiply(0.5 * deltaTau)))
					.add(Cov01_mp1.multiply(Dx01.scalarMultiply(deltaTau)))
					.add(Cov02_mp1.multiply(Dx02.scalarMultiply(deltaTau)))
					.add(Cov12_mp1.multiply(Dx12.scalarMultiply(deltaTau)));

			final RealMatrix F = Id.scalarMultiply(1.0 - r_m * deltaTau).add(driftTerm_m).add(diffTerm_m);
			final RealMatrix G = Id.scalarMultiply(1.0 + r_mp1 * deltaTau).subtract(driftTerm_mp1).subtract(diffTerm_mp1);

			final RealMatrix H = G.scalarMultiply(theta).add(Id.scalarMultiply(1.0 - theta));
			final RealMatrix A = F.scalarMultiply(1.0 - theta).add(Id.scalarMultiply(theta));

			RealMatrix rhs = A.multiply(U);

			for (int k2 = 0; k2 < n2; k2++) {
				for (int k1 = 0; k1 < n1; k1++) {
					for (int k0 = 0; k0 < n0; k0++) {
						final int idx = k0 + k1 * n0 + k2 * n0 * n1;
						if (!isBoundary[idx]) {
							continue;
						}

						final double x0 = x0Grid[k0];
						final double x1 = x1Grid[k1];
						final double x2 = x2Grid[k2];

						final double boundaryTime =
								spaceTimeDiscretization.getTimeDiscretization().getLastTime()
								- spaceTimeDiscretization.getTimeDiscretization().getTime(m + 1);

						final double[] lowerBounds = model.getValueAtLowerBoundary(product, boundaryTime, x0, x1, x2);
						final double[] upperBounds = model.getValueAtUpperBoundary(product, boundaryTime, x0, x1, x2);

						final double x0Lower = safeBound(lowerBounds, 0);
						final double x0Upper = safeBound(upperBounds, 0);
						final double x1Lower = safeBound(lowerBounds, 1);
						final double x1Upper = safeBound(upperBounds, 1);
						final double x2Lower = safeBound(lowerBounds, 2);
						final double x2Upper = safeBound(upperBounds, 2);

						double chosen = Double.NaN;
						if (k0 == 0) {
							chosen = x0Lower;
						} else if (k0 == n0 - 1) {
							chosen = x0Upper;
						} else if (k1 == 0) {
							chosen = x1Lower;
						} else if (k1 == n1 - 1) {
							chosen = x1Upper;
						} else if (k2 == 0) {
							chosen = x2Lower;
						} else if (k2 == n2 - 1) {
							chosen = x2Upper;
						}

						if (Double.isFinite(chosen)) {
							for (int col = 0; col < n; col++) {
								H.setEntry(idx, col, 0.0);
							}
							H.setEntry(idx, idx, 1.0);
							rhs.setEntry(idx, 0, chosen);
						}
					}
				}
			}

			if (exercise == ExerciseType.EUROPEAN) {
				final DecompositionSolver solver = new LUDecomposition(H).getSolver();
				U = solver.solve(rhs);
				z.setColumnMatrix(m + 1, U);
			}
			else if (exercise == ExerciseType.AMERICAN) {
				final double omega = 1.2;
				final SORDecomposition sor = new SORDecomposition(H);
				final RealMatrix zz = sor.getSol(U, rhs, omega, 500);

				for (int k2 = 0; k2 < n2; k2++) {
					for (int k1 = 0; k1 < n1; k1++) {
						for (int k0 = 0; k0 < n0; k0++) {
							final int idx = k0 + k1 * n0 + k2 * n0 * n1;
							final double payoff = valueAtMaturity.applyAsDouble(x0Grid[k0], x1Grid[k1], x2Grid[k2]);
							U.setEntry(idx, 0, Math.max(zz.getEntry(idx, 0), payoff));
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

	/**
	 * Returns the value at index idx from arr, or {@link Double#NaN} if arr is null or too short.
	 *
	 * @param arr Boundary array.
	 * @param idx Index to access.
	 * @return Boundary value or NaN if not available.
	 */
	private static double safeBound(final double[] arr, final int idx) {
		if (arr == null || idx < 0 || idx >= arr.length) {
			return Double.NaN;
		}
		return arr[idx];
	}

	/**
	 * Builds a block diagonal matrix with the given block repeated {@code numBlocks} times.
	 * Blocks are placed along the diagonal.
	 *
	 * @param block Block matrix.
	 * @param numBlocks Number of repetitions.
	 * @return Block diagonal matrix.
	 */
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

	/**
	 * Builds the Kronecker product {@code kron(A, B)} for banded matrices produced by
	 * {@link FiniteDifferenceMatrixBuilder}. This method assumes both {@code A} and {@code B} are at most 5-banded
	 * and iterates only over a small band.
	 *
	 * @param A Left matrix.
	 * @param B Right matrix.
	 * @return Kronecker product {@code kron(A, B)}.
	 */
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

	/**
	 * Builds {@code kron(A, I_n)} where {@code I_n} is the identity matrix of size {@code nIdentity}.
	 * Matrix {@code A} acts on the "slow" index.
	 *
	 * @param A Left matrix.
	 * @param nIdentity Size of the identity.
	 * @return Kronecker product {@code kron(A, I_nIdentity)}.
	 */
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

	/**
	 * Builds {@code kron(I_n, M)} where {@code I_n} is the identity matrix of size {@code nIdentity}.
	 * This is implemented as a block diagonal matrix with {@code M} repeated along the diagonal.
	 *
	 * @param M Matrix acting on the fast part.
	 * @param nIdentity Number of repetitions.
	 * @return {@code kron(I_nIdentity, M)}.
	 */
	private static RealMatrix buildKronWithIdentityRight(final RealMatrix M, final int nIdentity) {
		return buildBlockDiagonal(M, nIdentity);
	}
}