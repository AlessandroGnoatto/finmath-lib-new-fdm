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
 * <p>It supports models with state variables (x0, x1, x2) discretized on three grids including boundary points.
 * The time discretization is assumed to be the time-to-maturity discretization used throughout this project.</p>
 *
 * <p>Boundary conditions are enforced as Dirichlet conditions by overwriting the corresponding rows in the linear
 * system. A boundary row is overwritten only if the boundary provides a finite value for that dimension.
 * If a boundary returns {@link Double#NaN} (or does not provide an entry), the row is left intact so that
 * the PDE operator / stencil handles that boundary.</p>
 *
 * <p>The solver returns the full time history as a flattened matrix of dimension (n0 * n1 * n2) x (nT + 1).</p>
 *
 * <p>Flattening convention: index {@code k = i0 + i1*n0 + i2*n0*n1} where {@code i0} is the fastest index.</p>
 *
 * <p><b>Coefficient conventions</b> (consistent with {@link FDMThetaMethod2D}):
 * <ul>
 *   <li>The drift component for x0 is interpreted as a <i>percentage drift</i> and is multiplied by x0 in the operator.</li>
 *   <li>The factor loadings in the x0 row are interpreted as <i>percentage</i> loadings and are multiplied by x0 (and x0^2 for variance terms).</li>
 *   <li>Other state dimensions (x1, x2) are treated as unscaled state variables.</li>
 * </ul>
 * </p>
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

	/**
	 * Backward-compatible overload: terminal payoff depends on the first state variable only.
	 *
	 * @param time Maturity of the product.
	 * @param valueAtMaturity Terminal payoff as a function of the first state variable.
	 * @return Full time history on the flattened space-time grid: (n0*n1*n2) x (nT+1).
	 */
	@Override
	public double[][] getValues(final double time, final DoubleUnaryOperator valueAtMaturity) {
		return getValues(time, (x0, x1, x2) -> valueAtMaturity.applyAsDouble(x0));
	}

	/**
	 * Convenience overload: terminal payoff depends on the first two state variables.
	 *
	 * @param time Maturity of the product.
	 * @param valueAtMaturity Terminal payoff as a function of the first two state variables.
	 * @return Full time history on the flattened space-time grid: (n0*n1*n2) x (nT+1).
	 */
	public double[][] getValues(final double time, final DoubleBinaryOperator valueAtMaturity) {
		return getValues(time, (x0, x1, x2) -> valueAtMaturity.applyAsDouble(x0, x1));
	}

	/**
	 * Returns the full time history on the lifted 3D space-time grid using a payoff depending on all state variables.
	 *
	 * @param time Maturity of the product.
	 * @param valueAtMaturity Terminal payoff as a function of three state variables.
	 * @return Full time history on the flattened space-time grid: (n0*n1*n2) x (nT+1).
	 */
	public double[][] getValues(final double time, final DoubleTernaryOperator valueAtMaturity) {

		final Grid grid0Obj = spaceTimeDiscretization.getSpaceGrid(0);
		final Grid grid1Obj = spaceTimeDiscretization.getSpaceGrid(1);
		final Grid grid2Obj = spaceTimeDiscretization.getSpaceGrid(2);

		if(grid0Obj == null || grid1Obj == null || grid2Obj == null) {
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
		final int mSteps = spaceTimeDiscretization.getTimeDiscretization().getNumberOfTimeSteps();

		// 1D derivative matrices on FULL grids (including boundary points).
		final FiniteDifferenceMatrixBuilder builder0 = new FiniteDifferenceMatrixBuilder(x0Grid);
		final RealMatrix t10 = builder0.getFirstDerivativeMatrix();
		final RealMatrix t20 = builder0.getSecondDerivativeMatrix();

		final FiniteDifferenceMatrixBuilder builder1 = new FiniteDifferenceMatrixBuilder(x1Grid);
		final RealMatrix t11 = builder1.getFirstDerivativeMatrix();
		final RealMatrix t21 = builder1.getSecondDerivativeMatrix();

		final FiniteDifferenceMatrixBuilder builder2 = new FiniteDifferenceMatrixBuilder(x2Grid);
		final RealMatrix t12 = builder2.getFirstDerivativeMatrix();
		final RealMatrix t22 = builder2.getSecondDerivativeMatrix();

		/*
		 * Build 3D differential operators on flattened state (x0 fastest, then x1, then x2).
		 *
		 * D0  = I2 ⊗ I1 ⊗ T1(x0)
		 * D1  = I2 ⊗ T1(x1) ⊗ I0
		 * D2  = T1(x2) ⊗ I1 ⊗ I0
		 */
		final RealMatrix d0  = buildBlockDiagonal(t10, n1 * n2);
		final RealMatrix d00 = buildBlockDiagonal(t20, n1 * n2);

		final RealMatrix d1Inner  = buildKronWithIdentityLeft(t11, n0); // T1(x1) ⊗ I0
		final RealMatrix d11Inner = buildKronWithIdentityLeft(t21, n0); // T2(x1) ⊗ I0
		final RealMatrix d1  = buildBlockDiagonal(d1Inner, n2);         // I2 ⊗ (T1(x1) ⊗ I0)
		final RealMatrix d11 = buildBlockDiagonal(d11Inner, n2);

		final RealMatrix d2  = buildKronWithIdentityLeft(t12, n1 * n0); // T1(x2) ⊗ I_{n1*n0}
		final RealMatrix d22M = buildKronWithIdentityLeft(t22, n1 * n0); // T2(x2) ⊗ I_{n1*n0}

		/*
		 * Cross derivatives:
		 * D01 = I2 ⊗ (T1(x1) ⊗ T1(x0))
		 * D02 = T1(x2) ⊗ (I1 ⊗ T1(x0))
		 * D12 = T1(x2) ⊗ (T1(x1) ⊗ I0)
		 */
		final RealMatrix d01Inner = buildKron(t11, t10);                 // T1(x1) ⊗ T1(x0)
		final RealMatrix d01 = buildBlockDiagonal(d01Inner, n2);          // I2 ⊗ (T1(x1) ⊗ T1(x0))

		final RealMatrix i1KronD0 = buildBlockDiagonal(t10, n1);          // I1 ⊗ T1(x0)
		final RealMatrix d02 = buildKron(t12, i1KronD0);                  // T1(x2) ⊗ (I1 ⊗ T1(x0))

		final RealMatrix d12Inner = buildKronWithIdentityLeft(t11, n0);   // T1(x1) ⊗ I0
		final RealMatrix d12 = buildKron(t12, d12Inner);                  // T1(x2) ⊗ (T1(x1) ⊗ I0)

		// Diagonal matrices for x0 and x0^2 embedded in 3D as block diagonal.
		final RealMatrix d1x0 = buildBlockDiagonal(MatrixUtils.createRealDiagonalMatrix(x0Grid), n1 * n2);
		final double[] x0Sq = new double[n0];
		for(int i0 = 0; i0 < n0; i0++) {
			x0Sq[i0] = x0Grid[i0] * x0Grid[i0];
		}
		final RealMatrix d2x0 = buildBlockDiagonal(MatrixUtils.createRealDiagonalMatrix(x0Sq), n1 * n2);

		// Identify boundary nodes (Dirichlet enforcement candidates).
		final boolean[] isBoundary = new boolean[n];
		for(int i2 = 0; i2 < n2; i2++) {
			for(int i1 = 0; i1 < n1; i1++) {
				for(int i0 = 0; i0 < n0; i0++) {
					final int idx = i0 + i1 * n0 + i2 * n0 * n1;
					isBoundary[idx] = (i0 == 0 || i0 == n0 - 1 || i1 == 0 || i1 == n1 - 1 || i2 == 0 || i2 == n2 - 1);
				}
			}
		}

		// Initial condition at maturity (tau=0 <-> t=T): payoff on full grid.
		RealMatrix u = MatrixUtils.createRealMatrix(n, 1);
		for(int i2 = 0; i2 < n2; i2++) {
			for(int i1 = 0; i1 < n1; i1++) {
				for(int i0 = 0; i0 < n0; i0++) {
					final int idx = i0 + i1 * n0 + i2 * n0 * n1;
					u.setEntry(idx, 0, valueAtMaturity.applyAsDouble(x0Grid[i0], x1Grid[i1], x2Grid[i2]));
				}
			}
		}

		final RealMatrix z = MatrixUtils.createRealMatrix(n, timeLength);
		z.setColumnMatrix(0, u);

		// Time stepping backward in calendar time, forward in tau (time-to-maturity).
		for(int m = 0; m < mSteps; m++) {

			final double deltaTau = spaceTimeDiscretization.getTimeDiscretization().getTimeStep(m);

			// Convert tau-grid indices to running time t like in 1D/2D solvers.
			final double t_m = spaceTimeDiscretization.getTimeDiscretization().getTime(mSteps - m);
			final double t_mp1 = spaceTimeDiscretization.getTimeDiscretization().getTime(mSteps - (m + 1));

			final double tSafe_m = (t_m == 0.0 ? 1e-6 : t_m);
			final double tSafe_mp1 = (t_mp1 == 0.0 ? 1e-6 : t_mp1);

			// Risk free rates (consistent with 1D/2D solvers).
			final double rF_m = model.getRiskFreeCurve().getDiscountFactor(tSafe_m);
			final double r_m = -Math.log(rF_m) / tSafe_m;
			final double rF_mp1 = model.getRiskFreeCurve().getDiscountFactor(tSafe_mp1);
			final double r_mp1 = -Math.log(rF_mp1) / tSafe_mp1;

			// Percentage drift for x0, evaluated at first grid point for backward compatibility.
			final double mu0_m = model.getDrift(t_m, x0Grid[0], x1Grid[0], x2Grid[0])[0];
			final double mu0_mp1 = model.getDrift(t_mp1, x0Grid[0], x1Grid[0], x2Grid[0])[0];

			// Coefficient vectors (diagonals) at each node.
			final double[] mu1_m_vec = new double[n];
			final double[] mu1_mp1_vec = new double[n];
			final double[] mu2_m_vec = new double[n];
			final double[] mu2_mp1_vec = new double[n];

			// Diffusion / covariance coefficients.
			final double[] a00_m = new double[n];
			final double[] a00_mp1 = new double[n];

			final double[] a11_m = new double[n];
			final double[] a11_mp1 = new double[n];

			final double[] a22_m = new double[n];
			final double[] a22_mp1 = new double[n];

			final double[] a01_m = new double[n];
			final double[] a01_mp1 = new double[n];

			final double[] a02_m = new double[n];
			final double[] a02_mp1 = new double[n];

			final double[] a12_m = new double[n];
			final double[] a12_mp1 = new double[n];

			for(int i2 = 0; i2 < n2; i2++) {
				for(int i1 = 0; i1 < n1; i1++) {
					for(int i0 = 0; i0 < n0; i0++) {
						final int idx = i0 + i1 * n0 + i2 * n0 * n1;

						final double x0 = x0Grid[i0];
						final double x1 = x1Grid[i1];
						final double x2 = x2Grid[i2];

						final double[] drift_m = model.getDrift(t_m, x0, x1, x2);
						final double[] drift_mp1 = model.getDrift(t_mp1, x0, x1, x2);

						mu1_m_vec[idx] = drift_m.length > 1 ? drift_m[1] : 0.0;
						mu1_mp1_vec[idx] = drift_mp1.length > 1 ? drift_mp1[1] : 0.0;

						mu2_m_vec[idx] = drift_m.length > 2 ? drift_m[2] : 0.0;
						mu2_mp1_vec[idx] = drift_mp1.length > 2 ? drift_mp1[2] : 0.0;

						final double[][] b_m = model.getFactorLoading(t_m, x0, x1, x2);
						final double[][] b_mp1 = model.getFactorLoading(t_mp1, x0, x1, x2);

						double a00v_m = 0.0;
						double a11v_m = 0.0;
						double a22v_m = 0.0;
						double a01v_m = 0.0;
						double a02v_m = 0.0;
						double a12v_m = 0.0;

						double a00v_p = 0.0;
						double a11v_p = 0.0;
						double a22v_p = 0.0;
						double a01v_p = 0.0;
						double a02v_p = 0.0;
						double a12v_p = 0.0;

						final int nFactors_m = b_m[0].length;
						for(int f = 0; f < nFactors_m; f++) {
							final double b0 = b_m[0][f];
							final double b1 = b_m.length > 1 ? b_m[1][f] : 0.0;
							final double b2 = b_m.length > 2 ? b_m[2][f] : 0.0;

							a00v_m += b0 * b0;
							a11v_m += b1 * b1;
							a22v_m += b2 * b2;

							a01v_m += (x0 * b0) * b1;
							a02v_m += (x0 * b0) * b2;
							a12v_m += b1 * b2;
						}

						final int nFactors_p = b_mp1[0].length;
						for(int f = 0; f < nFactors_p; f++) {
							final double b0 = b_mp1[0][f];
							final double b1 = b_mp1.length > 1 ? b_mp1[1][f] : 0.0;
							final double b2 = b_mp1.length > 2 ? b_mp1[2][f] : 0.0;

							a00v_p += b0 * b0;
							a11v_p += b1 * b1;
							a22v_p += b2 * b2;

							a01v_p += (x0 * b0) * b1;
							a02v_p += (x0 * b0) * b2;
							a12v_p += b1 * b2;
						}

						a00_m[idx] = a00v_m;
						a00_mp1[idx] = a00v_p;

						a11_m[idx] = a11v_m;
						a11_mp1[idx] = a11v_p;

						a22_m[idx] = a22v_m;
						a22_mp1[idx] = a22v_p;

						a01_m[idx] = a01v_m;
						a01_mp1[idx] = a01v_p;

						a02_m[idx] = a02v_m;
						a02_mp1[idx] = a02v_p;

						a12_m[idx] = a12v_m;
						a12_mp1[idx] = a12v_p;
					}
				}
			}

			final RealMatrix mu1_m = MatrixUtils.createRealDiagonalMatrix(mu1_m_vec);
			final RealMatrix mu1_mp1 = MatrixUtils.createRealDiagonalMatrix(mu1_mp1_vec);
			final RealMatrix mu2_m = MatrixUtils.createRealDiagonalMatrix(mu2_m_vec);
			final RealMatrix mu2_mp1 = MatrixUtils.createRealDiagonalMatrix(mu2_mp1_vec);

			final RealMatrix a00m = MatrixUtils.createRealDiagonalMatrix(a00_m);
			final RealMatrix a00p = MatrixUtils.createRealDiagonalMatrix(a00_mp1);

			final RealMatrix a11m = MatrixUtils.createRealDiagonalMatrix(a11_m);
			final RealMatrix a11p = MatrixUtils.createRealDiagonalMatrix(a11_mp1);

			final RealMatrix a22m = MatrixUtils.createRealDiagonalMatrix(a22_m);
			final RealMatrix a22p = MatrixUtils.createRealDiagonalMatrix(a22_mp1);

			final RealMatrix a01m = MatrixUtils.createRealDiagonalMatrix(a01_m);
			final RealMatrix a01p = MatrixUtils.createRealDiagonalMatrix(a01_mp1);

			final RealMatrix a02m = MatrixUtils.createRealDiagonalMatrix(a02_m);
			final RealMatrix a02p = MatrixUtils.createRealDiagonalMatrix(a02_mp1);

			final RealMatrix a12m = MatrixUtils.createRealDiagonalMatrix(a12_m);
			final RealMatrix a12p = MatrixUtils.createRealDiagonalMatrix(a12_mp1);

			final RealMatrix ident = MatrixUtils.createRealIdentityMatrix(n);

			// Drift terms.
			final RealMatrix driftTerm_m =
					d1x0.scalarMultiply(mu0_m * deltaTau).multiply(d0)
					.add(mu1_m.scalarMultiply(deltaTau).multiply(d1))
					.add(mu2_m.scalarMultiply(deltaTau).multiply(d2));

			final RealMatrix driftTerm_mp1 =
					d1x0.scalarMultiply(mu0_mp1 * deltaTau).multiply(d0)
					.add(mu1_mp1.scalarMultiply(deltaTau).multiply(d1))
					.add(mu2_mp1.scalarMultiply(deltaTau).multiply(d2));

			// Diffusion / covariance terms.
			final RealMatrix diffTerm_m =
					a00m.multiply(d2x0.scalarMultiply(0.5 * deltaTau).multiply(d00))
					.add(a11m.multiply(d11.scalarMultiply(0.5 * deltaTau)))
					.add(a22m.multiply(d22M.scalarMultiply(0.5 * deltaTau)))
					.add(a01m.multiply(d01.scalarMultiply(deltaTau)))
					.add(a02m.multiply(d02.scalarMultiply(deltaTau)))
					.add(a12m.multiply(d12.scalarMultiply(deltaTau)));

			final RealMatrix diffTerm_mp1 =
					a00p.multiply(d2x0.scalarMultiply(0.5 * deltaTau).multiply(d00))
					.add(a11p.multiply(d11.scalarMultiply(0.5 * deltaTau)))
					.add(a22p.multiply(d22M.scalarMultiply(0.5 * deltaTau)))
					.add(a01p.multiply(d01.scalarMultiply(deltaTau)))
					.add(a02p.multiply(d02.scalarMultiply(deltaTau)))
					.add(a12p.multiply(d12.scalarMultiply(deltaTau)));

			final RealMatrix fMat = ident.scalarMultiply(1.0 - r_m * deltaTau).add(driftTerm_m).add(diffTerm_m);
			final RealMatrix gMat = ident.scalarMultiply(1.0 + r_mp1 * deltaTau).subtract(driftTerm_mp1).subtract(diffTerm_mp1);

			final RealMatrix hMat = gMat.scalarMultiply(theta).add(ident.scalarMultiply(1.0 - theta));
			final RealMatrix aMat = fMat.scalarMultiply(1.0 - theta).add(ident.scalarMultiply(theta));

			RealMatrix rhs = aMat.multiply(u);

			// Boundary enforcement (Dirichlet when boundary provides finite value).
			for(int i2 = 0; i2 < n2; i2++) {
				for(int i1 = 0; i1 < n1; i1++) {
					for(int i0 = 0; i0 < n0; i0++) {
						final int idx = i0 + i1 * n0 + i2 * n0 * n1;
						if(!isBoundary[idx]) {
							continue;
						}

						final double x0 = x0Grid[i0];
						final double x1 = x1Grid[i1];
						final double x2 = x2Grid[i2];

						final double boundaryTime =
								spaceTimeDiscretization.getTimeDiscretization().getLastTime()
								- spaceTimeDiscretization.getTimeDiscretization().getTime(m + 1);

						final double[] lowerBounds = model.getValueAtLowerBoundary(product, boundaryTime, x0, x1, x2);
						final double[] upperBounds = model.getValueAtUpperBoundary(product, boundaryTime, x0, x1, x2);

						final double b0Lower = safeBound(lowerBounds, 0);
						final double b0Upper = safeBound(upperBounds, 0);
						final double b1Lower = safeBound(lowerBounds, 1);
						final double b1Upper = safeBound(upperBounds, 1);
						final double b2Lower = safeBound(lowerBounds, 2);
						final double b2Upper = safeBound(upperBounds, 2);

						/*
						 * Choose which boundary value applies at this node.
						 * Priority at edges/corners: dimension 0, then 1, then 2.
						 */
						double chosen = Double.NaN;
						if(i0 == 0) {
							chosen = b0Lower;
						}
						else if(i0 == n0 - 1) {
							chosen = b0Upper;
						}
						else if(i1 == 0) {
							chosen = b1Lower;
						}
						else if(i1 == n1 - 1) {
							chosen = b1Upper;
						}
						else if(i2 == 0) {
							chosen = b2Lower;
						}
						else if(i2 == n2 - 1) {
							chosen = b2Upper;
						}

						if(Double.isFinite(chosen)) {
							for(int col = 0; col < n; col++) {
								hMat.setEntry(idx, col, 0.0);
							}
							hMat.setEntry(idx, idx, 1.0);
							rhs.setEntry(idx, 0, chosen);
						}
					}
				}
			}

			if(exercise == ExerciseType.EUROPEAN) {
				final DecompositionSolver solver = new LUDecomposition(hMat).getSolver();
				u = solver.solve(rhs);
				z.setColumnMatrix(m + 1, u);
			}
			else if(exercise == ExerciseType.AMERICAN) {
				final double omega = 1.2;
				final SORDecomposition sor = new SORDecomposition(hMat);
				final RealMatrix zz = sor.getSol(u, rhs, omega, 500);

				for(int i2 = 0; i2 < n2; i2++) {
					for(int i1 = 0; i1 < n1; i1++) {
						for(int i0 = 0; i0 < n0; i0++) {
							final int idx = i0 + i1 * n0 + i2 * n0 * n1;
							final double payoff = valueAtMaturity.applyAsDouble(x0Grid[i0], x1Grid[i1], x2Grid[i2]);
							u.setEntry(idx, 0, Math.max(zz.getEntry(idx, 0), payoff));
						}
					}
				}
				z.setColumnMatrix(m + 1, u);
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
	 * Returns the value at index {@code idx} from {@code arr}, or {@link Double#NaN} if {@code arr} is null or too short.
	 *
	 * @param arr Boundary array.
	 * @param idx Index to access.
	 * @return Boundary value or NaN if not available.
	 */
	private static double safeBound(final double[] arr, final int idx) {
		if(arr == null || idx < 0 || idx >= arr.length) {
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

		for(int b = 0; b < numBlocks; b++) {
			final int row0 = b * n;
			final int col0 = b * n;
			for(int i = 0; i < n; i++) {
				for(int j = Math.max(0, i - 2); j <= Math.min(n - 1, i + 2); j++) {
					final double v = block.getEntry(i, j);
					if(v != 0.0) {
						out.setEntry(row0 + i, col0 + j, v);
					}
				}
			}
		}
		return out;
	}

	/**
	 * Builds the Kronecker product {@code kron(A, B)} for banded matrices produced by {@link FiniteDifferenceMatrixBuilder}.
	 * This method assumes both {@code A} and {@code B} are at most 5-banded and iterates only over a small band.
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

		for(int i = 0; i < aR; i++) {
			for(int j = Math.max(0, i - 2); j <= Math.min(aC - 1, i + 2); j++) {
				final double a = A.getEntry(i, j);
				if(a == 0.0) {
					continue;
				}

				final int rowBase = i * bR;
				final int colBase = j * bC;

				for(int p = 0; p < bR; p++) {
					for(int q = Math.max(0, p - 2); q <= Math.min(bC - 1, p + 2); q++) {
						final double b = B.getEntry(p, q);
						if(b == 0.0) {
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

		for(int i = 0; i < aR; i++) {
			for(int j = Math.max(0, i - 2); j <= Math.min(aC - 1, i + 2); j++) {
				final double a = A.getEntry(i, j);
				if(a == 0.0) {
					continue;
				}

				for(int k = 0; k < nIdentity; k++) {
					out.setEntry(i * nIdentity + k, j * nIdentity + k, a);
				}
			}
		}
		return out;
	}
}