package net.finmath.finitedifference.solvers;

import java.util.function.DoubleUnaryOperator;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.DecompositionSolver;
import org.apache.commons.math3.linear.DiagonalMatrix;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.assetderivativevaluation.products.BarrierOption;
import net.finmath.finitedifference.assetderivativevaluation.products.EuropeanOption;
import net.finmath.finitedifference.assetderivativevaluation.products.FiniteDifferenceProduct;
import net.finmath.finitedifference.boundaries.BoundaryCondition;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.modelling.Exercise;
import net.finmath.modelling.products.BarrierType;
import net.finmath.modelling.products.CallOrPut;

/**
 * Direct 1D theta-method solver for European knock-in barrier options.
 *
 * <p>
 * The solver uses two states:
 * </p>
 * <ul>
 *   <li>an activated state, priced as the corresponding vanilla option on the full grid,</li>
 *   <li>a not-yet-hit state, solved directly on the same full grid.</li>
 * </ul>
 *
 * <p>
 * For the not-yet-hit state:
 * </p>
 * <ul>
 *   <li>nodes in the hit region are constrained to the activated-state value,</li>
 *   <li>the continuation-side far boundary is constrained to the discounted rebate,
 *       since for an IN option the rebate is paid at expiry if the barrier is never hit.</li>
 * </ul>
 *
 * <p>
 * This implementation currently supports only European exercise.
 * It is intended for barriers acting on the single 1D state variable.
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public class FDMThetaMethod1DDirectKnockIn {

	private static final double EPSILON = 1E-6;

	private final FiniteDifferenceEquityModel model;
	private final BarrierOption product;
	private final SpaceTimeDiscretization spaceTimeDiscretization;
	private final Exercise exercise;

	public FDMThetaMethod1DDirectKnockIn(
			final FiniteDifferenceEquityModel model,
			final BarrierOption product,
			final SpaceTimeDiscretization spaceTimeDiscretization,
			final Exercise exercise) {
		this.model = model;
		this.product = product;
		this.spaceTimeDiscretization = spaceTimeDiscretization;
		this.exercise = exercise;
	}

	public double[][] getValues(final double time) {

		if(!exercise.isEuropean()) {
			throw new IllegalArgumentException(
					"FDMThetaMethod1DDirectKnockIn currently supports only European exercise.");
		}

		if(product.getBarrierType() != BarrierType.DOWN_IN && product.getBarrierType() != BarrierType.UP_IN) {
			throw new IllegalArgumentException(
					"FDMThetaMethod1DDirectKnockIn supports only DOWN_IN and UP_IN.");
		}

		final double[] xGrid = spaceTimeDiscretization.getSpaceGrid(0).getGrid();
		final int nX = xGrid.length;

		final double theta = spaceTimeDiscretization.getTheta();

		final int timeLength = spaceTimeDiscretization.getTimeDiscretization().getNumberOfTimeSteps() + 1;
		final int M = spaceTimeDiscretization.getTimeDiscretization().getNumberOfTimeSteps();

		/*
		 * Step 1: solve the activated state on the full grid.
		 * Once the barrier has been hit, the product behaves like the corresponding vanilla option.
		 */
		final double[][] activatedSurface = solveActivatedState(time);

		final FiniteDifferenceMatrixBuilder fdBuilder = new FiniteDifferenceMatrixBuilder(xGrid);
		final RealMatrix T1 = fdBuilder.getFirstDerivativeMatrix();
		final RealMatrix T2 = fdBuilder.getSecondDerivativeMatrix();
		final RealMatrix I = MatrixUtils.createRealIdentityMatrix(nX);

		/*
		 * Step 2: initialize the not-yet-hit state at maturity.
		 *
		 * - if the maturity node is already in the hit region, activation occurs at maturity
		 *   and the value is the activated payoff,
		 * - otherwise the option never knocked in and pays the rebate at expiry.
		 */
		RealMatrix U = MatrixUtils.createRealMatrix(nX, 1);
		for(int i = 0; i < nX; i++) {
			if(isHitRegion(xGrid[i])) {
				U.setEntry(i, 0, activatedSurface[i][0]);
			}
			else {
				U.setEntry(i, 0, product.getRebate());
			}
		}

		final RealMatrix z = MatrixUtils.createRealMatrix(nX, timeLength);
		z.setColumnMatrix(0, U);

		for(int m = 0; m < M; m++) {

			final double deltaTau = spaceTimeDiscretization.getTimeDiscretization().getTimeStep(m);

			final double t_m = spaceTimeDiscretization.getTimeDiscretization().getTime(M - m);
			final double t_mp1 = spaceTimeDiscretization.getTimeDiscretization().getTime(M - (m + 1));

			final double tSafe_m = (t_m == 0.0 ? EPSILON : t_m);
			final double tSafe_mp1 = (t_mp1 == 0.0 ? EPSILON : t_mp1);

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

			/*
			 * Enforce direct knock-in constraints:
			 *
			 * - hit-region nodes are constrained to the activated-state value,
			 * - the continuation-side far boundary is constrained to the discounted rebate.
			 */
			for(int i = 0; i < nX; i++) {

				final boolean isLowerOuterBoundary = i == 0;
				final boolean isUpperOuterBoundary = i == nX - 1;

				final boolean isContinuationOuterBoundary =
						(product.getBarrierType() == BarrierType.DOWN_IN && isUpperOuterBoundary)
						|| (product.getBarrierType() == BarrierType.UP_IN && isLowerOuterBoundary);

				final boolean isHitNode = isHitRegion(xGrid[i]);

				if(isHitNode) {
					final double activatedValue = activatedSurface[i][m + 1];
					overwriteRowAsDirichlet(H, rhs, i, activatedValue);
				}
				else if(isContinuationOuterBoundary) {
					final double continuationBoundaryValue = getDiscountedRebate(boundaryTime);
					overwriteRowAsDirichlet(H, rhs, i, continuationBoundaryValue);
				}
			}

			final DecompositionSolver solver = new LUDecomposition(H).getSolver();
			U = solver.solve(rhs);

			/*
			 * Re-impose constraints after solve for numerical safety.
			 */
			for(int i = 0; i < nX; i++) {
				final boolean isLowerOuterBoundary = i == 0;
				final boolean isUpperOuterBoundary = i == nX - 1;

				final boolean isContinuationOuterBoundary =
						(product.getBarrierType() == BarrierType.DOWN_IN && isUpperOuterBoundary)
						|| (product.getBarrierType() == BarrierType.UP_IN && isLowerOuterBoundary);

				final boolean isHitNode = isHitRegion(xGrid[i]);

				if(isHitNode) {
					U.setEntry(i, 0, activatedSurface[i][m + 1]);
				}
				else if(isContinuationOuterBoundary) {
					U.setEntry(i, 0, getDiscountedRebate(boundaryTime));
				}
			}

			z.setColumnMatrix(m + 1, U);
		}

		return z.getData();
	}

	public double[] getValue(final double evaluationTime, final double time) {
		final RealMatrix values = new Array2DRowRealMatrix(getValues(time));
		final double tau = time - evaluationTime;
		final int timeIndex =
				spaceTimeDiscretization.getTimeDiscretization().getTimeIndexNearestLessOrEqual(tau);
		return values.getColumn(timeIndex);
	}

	private double[][] solveActivatedState(final double time) {

		final EuropeanOption activatedVanilla =
				new EuropeanOption(
						product.getUnderlyingName(),
						product.getMaturity(),
						product.getStrike(),
						product.getCallOrPut()
				);

		final FDMSolver vanillaSolver =
				new FDMThetaMethod1D(
						model,
						activatedVanilla,
						spaceTimeDiscretization,
						exercise
				);

		final DoubleUnaryOperator vanillaPayoff;
		if(product.getCallOrPut() == CallOrPut.CALL) {
			vanillaPayoff = x -> Math.max(x - product.getStrike(), 0.0);
		}
		else {
			vanillaPayoff = x -> Math.max(product.getStrike() - x, 0.0);
		}

		return vanillaSolver.getValues(time, vanillaPayoff);
	}

	private boolean isHitRegion(final double x) {
		if(product.getBarrierType() == BarrierType.DOWN_IN) {
			return x <= product.getBarrierValue();
		}
		else if(product.getBarrierType() == BarrierType.UP_IN) {
			return x >= product.getBarrierValue();
		}
		else {
			throw new IllegalArgumentException("Unsupported barrier type for direct knock-in solver.");
		}
	}

	private double getDiscountedRebate(double time) {
		time = Math.max(time, EPSILON);

		final double discountFactorRiskFree = model.getRiskFreeCurve().getDiscountFactor(time);
		final double riskFreeRate = -Math.log(discountFactorRiskFree) / time;

		return product.getRebate() * Math.exp(-riskFreeRate * (product.getMaturity() - time));
	}

	private static void overwriteRowAsDirichlet(
			final RealMatrix H,
			final RealMatrix rhs,
			final int rowIndex,
			final double value) {

		final int n = H.getColumnDimension();
		for(int col = 0; col < n; col++) {
			H.setEntry(rowIndex, col, 0.0);
		}
		H.setEntry(rowIndex, rowIndex, 1.0);
		rhs.setEntry(rowIndex, 0, value);
	}
}