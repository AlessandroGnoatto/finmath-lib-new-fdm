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
import net.finmath.finitedifference.assetderivativevaluation.products.BarrierOption;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.modelling.Exercise;
import net.finmath.modelling.products.BarrierType;

/**
 * Direct two-state theta-method solver for 2D knock-in barrier options.
 *
 * <p>
 * Regime 0 = not yet activated (barrier not yet hit).
 * Regime 1 = already activated (barrier has been hit).
 * </p>
 *
 * <p>
 * Both regimes evolve under the same 2D PDE operator on the same full spatial grid.
 * The active regime uses the ordinary vanilla PDE and caller-supplied outer boundary conditions.
 * The inactive regime is coupled to the active regime on the whole already-hit region
 * with respect to the first state variable.
 * </p>
 *
 * <p>
 * This is a first-cut European-only implementation intended for Heston-style 2D models.
 * </p>
 */
public class FDMThetaMethod2DTwoState implements FDMSolver {

	private static final double EPSILON = 1E-10;

	private final FiniteDifferenceEquityModel model;
	private final BarrierOption product;
	private final SpaceTimeDiscretization spaceTimeDiscretization;
	private final Exercise exercise;
	private final TwoStateActiveBoundaryProvider2D activeBoundaryProvider;

	public FDMThetaMethod2DTwoState(
			final FiniteDifferenceEquityModel model,
			final BarrierOption product,
			final SpaceTimeDiscretization spaceTimeDiscretization,
			final Exercise exercise,
			final TwoStateActiveBoundaryProvider2D activeBoundaryProvider) {
		this.model = model;
		this.product = product;
		this.spaceTimeDiscretization = spaceTimeDiscretization;
		this.exercise = exercise;
		this.activeBoundaryProvider = activeBoundaryProvider;
	}

	@Override
	public double[][] getValues(final double time, final DoubleUnaryOperator valueAtMaturity) {
		return getValues(time, (x0, x1) -> valueAtMaturity.applyAsDouble(x0));
	}

	public double[][] getValues(final double time, final DoubleBinaryOperator valueAtMaturity) {

		if(!exercise.isEuropean()) {
			throw new IllegalArgumentException("FDMThetaMethod2DTwoState currently supports only European exercise.");
		}

		if(activeBoundaryProvider == null) {
			throw new IllegalArgumentException("Active boundary provider must not be null.");
		}

		final BarrierType barrierType = product.getBarrierType();
		if(barrierType != BarrierType.DOWN_IN && barrierType != BarrierType.UP_IN) {
			throw new IllegalArgumentException("FDMThetaMethod2DTwoState is only for knock-in barrier options.");
		}

		final Grid x0GridObj = spaceTimeDiscretization.getSpaceGrid(0);
		final Grid x1GridObj = spaceTimeDiscretization.getSpaceGrid(1);
		if(x0GridObj == null || x1GridObj == null) {
			throw new IllegalArgumentException(
					"SpaceTimeDiscretization must provide two space grids (dimension 0 and dimension 1).");
		}

		final double[] x0Grid = x0GridObj.getGrid();
		final double[] x1Grid = x1GridObj.getGrid();

		validateBarrierOnGridNode(x0Grid, product.getBarrierValue());

		final int n0 = x0Grid.length;
		final int n1 = x1Grid.length;
		final int n = n0 * n1;
		final int nTwoState = 2 * n;

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

		RealMatrix UInactive = MatrixUtils.createRealMatrix(n, 1);
		RealMatrix UActive = MatrixUtils.createRealMatrix(n, 1);

		for(int j = 0; j < n1; j++) {
			for(int i = 0; i < n0; i++) {
				final int k = flatten(i, j, n0);
				final double x0 = x0Grid[i];
				final double x1 = x1Grid[j];
				final double payoff = valueAtMaturity.applyAsDouble(x0, x1);

				UActive.setEntry(k, 0, payoff);

				if(isAlreadyHitRegion(x0, barrierType, product.getBarrierValue())) {
					UInactive.setEntry(k, 0, payoff);
				}
				else {
					UInactive.setEntry(k, 0, product.getRebate());
				}
			}
		}

		RealMatrix U = MatrixUtils.createRealMatrix(nTwoState, 1);
		copyBlockIntoVector(UInactive, U, 0);
		copyBlockIntoVector(UActive, U, n);

		final RealMatrix z = MatrixUtils.createRealMatrix(n, timeLength);
		z.setColumnMatrix(0, UInactive);

		for(int m = 0; m < M; m++) {

			final double deltaTau = spaceTimeDiscretization.getTimeDiscretization().getTimeStep(m);

			final double t_m = spaceTimeDiscretization.getTimeDiscretization().getTime(M - m);
			final double t_mp1 = spaceTimeDiscretization.getTimeDiscretization().getTime(M - (m + 1));

			final double tSafe_m = Math.max(t_m, 1E-6);
			final double tSafe_mp1 = Math.max(t_mp1, 1E-6);

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

			for(int j = 0; j < n1; j++) {
				for(int i = 0; i < n0; i++) {
					final int k = flatten(i, j, n0);
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

					for(int f = 0; f < b_m[0].length; f++) {
						final double b0f = b_m[0][f];
						final double b1f = b_m.length > 1 ? b_m[1][f] : 0.0;
						a00v_m += b0f * b0f;
						a11v_m += b1f * b1f;
						a01v_m += b0f * b1f;
					}

					double a00v_p = 0.0;
					double a11v_p = 0.0;
					double a01v_p = 0.0;

					for(int f = 0; f < b_mp1[0].length; f++) {
						final double b0f = b_mp1[0][f];
						final double b1f = b_mp1.length > 1 ? b_mp1[1][f] : 0.0;
						a00v_p += b0f * b0f;
						a11v_p += b1f * b1f;
						a01v_p += b0f * b1f;
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
			final RealMatrix I2 = MatrixUtils.createRealIdentityMatrix(nTwoState);

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

			final RealMatrix F2D = I.scalarMultiply(1.0 - r_m * deltaTau).add(driftTerm_m).add(diffTerm_m);
			final RealMatrix G2D = I.scalarMultiply(1.0 + r_mp1 * deltaTau).subtract(driftTerm_mp1).subtract(diffTerm_mp1);

			final RealMatrix FBlock = MatrixUtils.createRealMatrix(nTwoState, nTwoState);
			final RealMatrix GBlock = MatrixUtils.createRealMatrix(nTwoState, nTwoState);

			FBlock.setSubMatrix(F2D.getData(), 0, 0);
			FBlock.setSubMatrix(F2D.getData(), n, n);

			GBlock.setSubMatrix(G2D.getData(), 0, 0);
			GBlock.setSubMatrix(G2D.getData(), n, n);

			RealMatrix H = GBlock.scalarMultiply(theta).add(I2.scalarMultiply(1.0 - theta));
			final RealMatrix A = FBlock.scalarMultiply(1.0 - theta).add(I2.scalarMultiply(theta));

			RealMatrix rhs = A.multiply(U);

			final double tau_mp1 = spaceTimeDiscretization.getTimeDiscretization().getTime(m + 1);
			final double currentTime = spaceTimeDiscretization.getTimeDiscretization().getLastTime() - tau_mp1;

			applyOuterBoundaryConditions(H, rhs, currentTime, x0Grid, x1Grid, n0, n1, barrierType, n);

			for(int j = 0; j < n1; j++) {
				for(int i = 0; i < n0; i++) {
					final int k = flatten(i, j, n0);
					final double x0 = x0Grid[i];

					if(isAlreadyHitRegion(x0, barrierType, product.getBarrierValue())) {
						overwriteCouplingRow(H, rhs, k, n);
					}
				}
			}

			final DecompositionSolver solver = new LUDecomposition(H).getSolver();
			U = solver.solve(rhs);

			for(int j = 0; j < n1; j++) {
				for(int i = 0; i < n0; i++) {
					final int k = flatten(i, j, n0);
					final double x0 = x0Grid[i];

					if(isAlreadyHitRegion(x0, barrierType, product.getBarrierValue())) {
						U.setEntry(k, 0, U.getEntry(n + k, 0));
					}
				}
			}

			UInactive = U.getSubMatrix(0, n - 1, 0, 0);
			UActive = U.getSubMatrix(n, 2 * n - 1, 0, 0);

			z.setColumnMatrix(m + 1, UInactive);
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

	private void applyOuterBoundaryConditions(
			final RealMatrix H,
			final RealMatrix rhs,
			final double currentTime,
			final double[] x0Grid,
			final double[] x1Grid,
			final int n0,
			final int n1,
			final BarrierType barrierType,
			final int n) {

		final double discountedNoHitValue = getDiscountedNoHitValue(currentTime);

		for(int j = 0; j < n1; j++) {
			for(int i = 0; i < n0; i++) {
				final int k = flatten(i, j, n0);
				final double x0 = x0Grid[i];
				final double x1 = x1Grid[j];

				final double[] lowerActive = activeBoundaryProvider.getLowerBoundaryValues(currentTime, x0, x1);
				final double[] upperActive = activeBoundaryProvider.getUpperBoundaryValues(currentTime, x0, x1);

				/*
				 * Active regime boundaries.
				 */
				if(i == 0 && !Double.isNaN(lowerActive[0])) {
					overwriteDirichletRow(H, rhs, n + k, lowerActive[0]);
				}
				if(i == n0 - 1 && !Double.isNaN(upperActive[0])) {
					overwriteDirichletRow(H, rhs, n + k, upperActive[0]);
				}
				if(j == 0 && !Double.isNaN(lowerActive[1])) {
					overwriteDirichletRow(H, rhs, n + k, lowerActive[1]);
				}
				if(j == n1 - 1 && !Double.isNaN(upperActive[1])) {
					overwriteDirichletRow(H, rhs, n + k, upperActive[1]);
				}

				/*
				 * Inactive regime:
				 * on the spot continuation-side outer boundary, use discounted no-hit value.
				 * On the spot hit-side outer boundary, couple to active regime.
				 *
				 * Variance boundaries are currently left to the PDE rows.
				 */
				if(barrierType == BarrierType.DOWN_IN) {
					if(i == 0) {
						overwriteCouplingRow(H, rhs, k, n);
					}
					else if(i == n0 - 1) {
						overwriteDirichletRow(H, rhs, k, discountedNoHitValue);
					}
				}
				else if(barrierType == BarrierType.UP_IN) {
					if(i == 0) {
						overwriteDirichletRow(H, rhs, k, discountedNoHitValue);
					}
					else if(i == n0 - 1) {
						overwriteCouplingRow(H, rhs, k, n);
					}
				}
			}
		}
	}

	private double getDiscountedNoHitValue(final double currentTime) {
		if(product.getRebate() == 0.0) {
			return 0.0;
		}

		final double t = Math.max(currentTime, EPSILON);
		final double maturity = product.getMaturity();

		if(t >= maturity) {
			return product.getRebate();
		}

		final double dfNow = model.getRiskFreeCurve().getDiscountFactor(t);
		final double dfMat = model.getRiskFreeCurve().getDiscountFactor(maturity);

		return product.getRebate() * dfMat / dfNow;
	}

	private void overwriteDirichletRow(
			final RealMatrix H,
			final RealMatrix rhs,
			final int row,
			final double value) {

		for(int col = 0; col < H.getColumnDimension(); col++) {
			H.setEntry(row, col, 0.0);
		}
		H.setEntry(row, row, 1.0);
		rhs.setEntry(row, 0, value);
	}

	private void overwriteCouplingRow(
			final RealMatrix H,
			final RealMatrix rhs,
			final int inactiveIndex,
			final int n) {

		for(int col = 0; col < H.getColumnDimension(); col++) {
			H.setEntry(inactiveIndex, col, 0.0);
		}

		H.setEntry(inactiveIndex, inactiveIndex, 1.0);
		H.setEntry(inactiveIndex, n + inactiveIndex, -1.0);
		rhs.setEntry(inactiveIndex, 0, 0.0);
	}

	private void copyBlockIntoVector(final RealMatrix block, final RealMatrix vector, final int rowOffset) {
		for(int i = 0; i < block.getRowDimension(); i++) {
			vector.setEntry(rowOffset + i, 0, block.getEntry(i, 0));
		}
	}

	private void validateBarrierOnGridNode(final double[] grid, final double barrier) {
		for(final double x : grid) {
			if(Math.abs(x - barrier) < 1E-12) {
				return;
			}
		}

		throw new IllegalArgumentException(
				"Barrier must coincide with a first-state-variable grid node for direct two-state knock-in pricing.");
	}

	private boolean isAlreadyHitRegion(
			final double x0,
			final BarrierType barrierType,
			final double barrier) {

		switch(barrierType) {
		case DOWN_IN:
			return x0 <= barrier;
		case UP_IN:
			return x0 >= barrier;
		default:
			throw new IllegalArgumentException("Unsupported barrier type: " + barrierType);
		}
	}

	private static int flatten(final int i0, final int i1, final int n0) {
		return i0 + i1 * n0;
	}

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