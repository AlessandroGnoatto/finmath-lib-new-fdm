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
 * Constructs and solves matrix systems for the theta method applied to two-dimensional PDEs using finite differences.
 *
 * <p>This solver follows the structure of {@link FDMThetaMethod1D} as closely as possible: it uses
 * {@link FiniteDifferenceEquityModel#getDrift(double, double...)} and
 * {@link FiniteDifferenceEquityModel#getFactorLoading(double, double...)} to obtain the PDE coefficients.</p>
 *
 * <p>It supports generic stochastic-volatility-type models with state variables (x0, x1) discretized on two grids
 * including boundary points. The time discretization is assumed to be the time-to-maturity discretization used
 * throughout this project.</p>
 *
 * <p>Boundary conditions are enforced as Dirichlet conditions by overwriting the corresponding rows in the linear
 * system, keeping the numerical scheme consistent with the prescribed boundary values.</p>
 *
 * <p>The solver returns the full time history as a flattened matrix of dimension (n0 * n1) x (nT + 1).</p>
 *
 * @author Enrico De Vecchi
 * @author Alessandro Gnoatto
 */
public class FDMThetaMethod2D implements FDMSolver {

	private final FiniteDifferenceEquityModel model;
	private final FiniteDifferenceProduct product;
	private final SpaceTimeDiscretization spaceTimeDiscretization;
	private final ExerciseType exercise;

	/**
	 * Creates a 2D theta method solver.
	 *
	 * @param model Finite difference equity model providing drift, factor loadings, and boundary conditions.
	 * @param product Product used for boundary value queries.
	 * @param spaceTimeDiscretization Space-time discretization (two space grids + time discretization).
	 * @param exercise Exercise type (European/American).
	 */
	public FDMThetaMethod2D(
			final FiniteDifferenceEquityModel model,
			final FiniteDifferenceProduct product,
			final SpaceTimeDiscretization spaceTimeDiscretization,
			final ExerciseType exercise) {
		this.model = model;
		this.product = product;
		this.spaceTimeDiscretization = spaceTimeDiscretization;
		this.exercise = exercise;
	}

	/**
	 * Backward-compatible overload: terminal payoff depends on the first state variable only.
	 *
	 * @param time Maturity of the product.
	 * @param valueAtMaturity Terminal payoff as a function of the first state variable.
	 * @return Full time history on the flattened space-time grid: (n0*n1) x (nT+1).
	 */
	@Override
	public double[][] getValues(final double time, final DoubleUnaryOperator valueAtMaturity) {
		return getValues(time, (x0, x1) -> valueAtMaturity.applyAsDouble(x0));
	}

	/**
	 * Returns the full time history on the lifted 2D space-time grid using a payoff depending on both state variables.
	 *
	 * <p>This overload is intended for Markov-lifted products (e.g., Asian options) where the terminal payoff depends
	 * on the second state variable (e.g., the running integral).</p>
	 *
	 * @param time Maturity of the product.
	 * @param valueAtMaturity Terminal payoff as a function of both state variables.
	 * @return Full time history on the flattened space-time grid: (n0*n1) x (nT+1).
	 */
	public double[][] getValues(final double time, final DoubleBinaryOperator valueAtMaturity) {

		final Grid sGridObj = spaceTimeDiscretization.getSpaceGrid(0);
		final Grid vGridObj = spaceTimeDiscretization.getSpaceGrid(1);
		if (sGridObj == null || vGridObj == null) {
			throw new IllegalArgumentException(
					"SpaceTimeDiscretization must provide two space grids (dimension 0 and dimension 1).");
		}

		final double[] sGrid = sGridObj.getGrid();
		final double[] vGrid = vGridObj.getGrid();

		final int nS = sGrid.length;
		final int nV = vGrid.length;
		final int n = nS * nV;

		final double theta = spaceTimeDiscretization.getTheta();

		final int timeLength = spaceTimeDiscretization.getTimeDiscretization().getNumberOfTimeSteps() + 1;
		final int M = spaceTimeDiscretization.getTimeDiscretization().getNumberOfTimeSteps();

		// Build 1D derivative matrices on the FULL grids (including boundary points).
		final FiniteDifferenceMatrixBuilder sBuilder = new FiniteDifferenceMatrixBuilder(sGrid);
		final RealMatrix T1S = sBuilder.getFirstDerivativeMatrix();
		final RealMatrix T2S = sBuilder.getSecondDerivativeMatrix();

		final FiniteDifferenceMatrixBuilder vBuilder = new FiniteDifferenceMatrixBuilder(vGrid);
		final RealMatrix T1V = vBuilder.getFirstDerivativeMatrix();
		final RealMatrix T2V = vBuilder.getSecondDerivativeMatrix();

		// Build 2D differential operators on flattened state (S fastest, then v).
		final RealMatrix DS = buildBlockDiagonal(T1S, nV);               // d/dS
		final RealMatrix DSS = buildBlockDiagonal(T2S, nV);              // d2/dS2
		final RealMatrix DV = buildKronWithIdentityLeft(T1V, nS);        // d/dv
		final RealMatrix DVV = buildKronWithIdentityLeft(T2V, nS);       // d2/dv2
		final RealMatrix DSV = buildKron(T1V, T1S);                      // d2/(dv dS)

		// Diagonal matrices for S and S^2 (embedded in 2D as block diagonal).
		final RealMatrix D1S = buildBlockDiagonal(MatrixUtils.createRealDiagonalMatrix(sGrid), nV);
		final double[] s2 = new double[nS];
		for (int i = 0; i < nS; i++) {
			s2[i] = sGrid[i] * sGrid[i];
		}
		final RealMatrix D2S = buildBlockDiagonal(MatrixUtils.createRealDiagonalMatrix(s2), nV);

		// Identify boundary nodes (Dirichlet enforcement).
		final boolean[] isBoundary = new boolean[n];
		for (int j = 0; j < nV; j++) {
			for (int i = 0; i < nS; i++) {
				final int k = i + j * nS;
				isBoundary[k] = (i == 0 || i == nS - 1 || j == 0 || j == nV - 1);
			}
		}

		// Initial condition at maturity (tau=0 <-> t=T): payoff on full (S,v) grid.
		RealMatrix U = MatrixUtils.createRealMatrix(n, 1);
		for (int j = 0; j < nV; j++) {
			for (int i = 0; i < nS; i++) {
				final int k = i + j * nS;
				U.setEntry(k, 0, valueAtMaturity.applyAsDouble(sGrid[i], vGrid[j]));
			}
		}

		final RealMatrix z = MatrixUtils.createRealMatrix(n, timeLength);
		z.setColumnMatrix(0, U);

		// Time stepping backward in calendar time, forward in tau (time-to-maturity).
		for (int m = 0; m < M; m++) {

			final double deltaTau = spaceTimeDiscretization.getTimeDiscretization().getTimeStep(m);

			// Convert tau-grid indices to running time t like in FDMThetaMethod1D.
			final double t_m = spaceTimeDiscretization.getTimeDiscretization().getTime(M - m);
			final double t_mp1 = spaceTimeDiscretization.getTimeDiscretization().getTime(M - (m + 1));

			final double tSafe_m = (t_m == 0.0 ? 1e-6 : t_m);
			final double tSafe_mp1 = (t_mp1 == 0.0 ? 1e-6 : t_mp1);

			// Risk free rates (consistent with FDMThetaMethod1D implementation).
			final double rF_m = model.getRiskFreeCurve().getDiscountFactor(tSafe_m);
			final double r_m = -Math.log(rF_m) / tSafe_m;
			final double rF_mp1 = model.getRiskFreeCurve().getDiscountFactor(tSafe_mp1);
			final double r_mp1 = -Math.log(rF_mp1) / tSafe_mp1;

			// Drift in S is treated as percentage drift (to be multiplied by S via D1S), consistent with 1D solver.
			// For general SV models, evaluate at first grid point like in the 1D solver.
			final double muS_m = model.getDrift(t_m, sGrid[0], vGrid[0])[0];
			final double muS_mp1 = model.getDrift(t_mp1, sGrid[0], vGrid[0])[0];

			// Build diagonal coefficient vectors for v drift and for covariance terms at each node.
			final double[] muV_m_vec = new double[n];
			final double[] muV_mp1_vec = new double[n];

			final double[] aSS_m = new double[n];    // sum_k bS,k^2 (percentage variance for S)
			final double[] aSS_mp1 = new double[n];

			final double[] aVV_m = new double[n];    // sum_k bV,k^2 (variance variance)
			final double[] aVV_mp1 = new double[n];

			final double[] aSV_m = new double[n];    // sum_k (S*bS,k)*bV,k  (includes S scaling)
			final double[] aSV_mp1 = new double[n];

			for (int j = 0; j < nV; j++) {
				for (int i = 0; i < nS; i++) {
					final int k = i + j * nS;
					final double S = sGrid[i];
					final double v = vGrid[j];

					final double[] drift_m = model.getDrift(t_m, S, v);
					final double[] drift_mp1 = model.getDrift(t_mp1, S, v);
					muV_m_vec[k] = drift_m.length > 1 ? drift_m[1] : 0.0;
					muV_mp1_vec[k] = drift_mp1.length > 1 ? drift_mp1[1] : 0.0;

					final double[][] b_m = model.getFactorLoading(t_m, S, v);
					final double[][] b_mp1 = model.getFactorLoading(t_mp1, S, v);

					double aSSm = 0.0;
					double aSSp = 0.0;
					double aVVm = 0.0;
					double aVVp = 0.0;
					double aSVm = 0.0;
					double aSVp = 0.0;

					final int nFactors_m = b_m[0].length;
					for (int f = 0; f < nFactors_m; f++) {
						final double bS = b_m[0][f];
						final double bV = b_m.length > 1 ? b_m[1][f] : 0.0;
						aSSm += bS * bS;
						aVVm += bV * bV;
						aSVm += (S * bS) * bV;
					}

					final int nFactors_p = b_mp1[0].length;
					for (int f = 0; f < nFactors_p; f++) {
						final double bS = b_mp1[0][f];
						final double bV = b_mp1.length > 1 ? b_mp1[1][f] : 0.0;
						aSSp += bS * bS;
						aVVp += bV * bV;
						aSVp += (S * bS) * bV;
					}

					aSS_m[k] = aSSm;
					aSS_mp1[k] = aSSp;
					aVV_m[k] = aVVm;
					aVV_mp1[k] = aVVp;
					aSV_m[k] = aSVm;
					aSV_mp1[k] = aSVp;
				}
			}

			final RealMatrix MuV_m = MatrixUtils.createRealDiagonalMatrix(muV_m_vec);
			final RealMatrix MuV_mp1 = MatrixUtils.createRealDiagonalMatrix(muV_mp1_vec);

			final RealMatrix A_SS_m = MatrixUtils.createRealDiagonalMatrix(aSS_m);
			final RealMatrix A_SS_mp1 = MatrixUtils.createRealDiagonalMatrix(aSS_mp1);
			final RealMatrix A_VV_m = MatrixUtils.createRealDiagonalMatrix(aVV_m);
			final RealMatrix A_VV_mp1 = MatrixUtils.createRealDiagonalMatrix(aVV_mp1);
			final RealMatrix A_SV_m = MatrixUtils.createRealDiagonalMatrix(aSV_m);
			final RealMatrix A_SV_mp1 = MatrixUtils.createRealDiagonalMatrix(aSV_mp1);

			final RealMatrix I = MatrixUtils.createRealIdentityMatrix(n);

			final RealMatrix driftTerm_m =
					D1S.scalarMultiply(muS_m * deltaTau).multiply(DS)
					.add(MuV_m.scalarMultiply(deltaTau).multiply(DV));

			final RealMatrix driftTerm_mp1 =
					D1S.scalarMultiply(muS_mp1 * deltaTau).multiply(DS)
					.add(MuV_mp1.scalarMultiply(deltaTau).multiply(DV));

			final RealMatrix diffTerm_m =
					A_SS_m.multiply(D2S.scalarMultiply(0.5 * deltaTau).multiply(DSS))
					.add(A_VV_m.multiply(DVV.scalarMultiply(0.5 * deltaTau)))
					.add(A_SV_m.multiply(DSV.scalarMultiply(deltaTau)));

			final RealMatrix diffTerm_mp1 =
					A_SS_mp1.multiply(D2S.scalarMultiply(0.5 * deltaTau).multiply(DSS))
					.add(A_VV_mp1.multiply(DVV.scalarMultiply(0.5 * deltaTau)))
					.add(A_SV_mp1.multiply(DSV.scalarMultiply(deltaTau)));

			final RealMatrix F = I.scalarMultiply(1.0 - r_m * deltaTau).add(driftTerm_m).add(diffTerm_m);
			final RealMatrix G = I.scalarMultiply(1.0 + r_mp1 * deltaTau).subtract(driftTerm_mp1).subtract(diffTerm_mp1);

			final RealMatrix H = G.scalarMultiply(theta).add(I.scalarMultiply(1.0 - theta));
			final RealMatrix A = F.scalarMultiply(1.0 - theta).add(I.scalarMultiply(theta));

			RealMatrix rhs = A.multiply(U);

			for (int j = 0; j < nV; j++) {
				for (int i = 0; i < nS; i++) {
					final int k = i + j * nS;
					if (!isBoundary[k]) {
						continue;
					}

					final double S = sGrid[i];
					final double v = vGrid[j];

					final double boundaryTime =
							spaceTimeDiscretization.getTimeDiscretization().getLastTime()
							- spaceTimeDiscretization.getTimeDiscretization().getTime(m + 1);

					final double[] lowerBounds = model.getValueAtLowerBoundary(product, boundaryTime, S, v);
					final double[] upperBounds = model.getValueAtUpperBoundary(product, boundaryTime, S, v);

					// helper to safely read bounds; returns NaN when not provided
					final double sLower = safeBound(lowerBounds, 0);
					final double sUpper = safeBound(upperBounds, 0);
					final double vLower = safeBound(lowerBounds, 1);
					final double vUpper = safeBound(upperBounds, 1);

					/*
					 * Choose which boundary value applies at this node.
					 * If we are at a corner (two boundaries), prioritize S-bound for backward compatibility.
					 *
					 * Note: we *only* enforce the row if the chosen boundaryValue is finite.
					 * If it is NaN (i.e. boundary does not provide a Dirichlet for that dimension),
					 * we leave the row as-is (so the PDE operator / upwind stencil handles it).
					 */
					double chosenBoundaryValue = Double.NaN;
					if (i == 0) {
						chosenBoundaryValue = sLower;
					} else if (i == nS - 1) {
						chosenBoundaryValue = sUpper;
					} else if (j == 0) {
						chosenBoundaryValue = vLower;
					} else if (j == nV - 1) {
						chosenBoundaryValue = vUpper;
					} else {
						// should not happen because isBoundary[k] must be true
						chosenBoundaryValue = sLower;
					}

					if (Double.isFinite(chosenBoundaryValue)) {
						// enforce Dirichlet only when we have a finite boundary value
						for (int col = 0; col < n; col++) {
							H.setEntry(k, col, 0.0);
						}
						H.setEntry(k, k, 1.0);
						rhs.setEntry(k, 0, chosenBoundaryValue);
					}
					// else: do nothing — leave the linear system row intact
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

				for (int j = 0; j < nV; j++) {
					for (int i = 0; i < nS; i++) {
						final int k = i + j * nS;
						final double payoff = valueAtMaturity.applyAsDouble(sGrid[i], vGrid[j]);
						U.setEntry(k, 0, Math.max(zz.getEntry(k, 0), payoff));
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
	 * Returns the value at index idx from arr, or Double.NaN if arr is null or too short.
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
}