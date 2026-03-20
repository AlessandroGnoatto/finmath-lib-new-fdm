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
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.modelling.Exercise;
import net.finmath.modelling.products.BarrierType;

/**
 * Direct two-state theta-method solver for 1D knock-in barrier options.
 *
 * <p>
 * Regime 0 = not yet activated (barrier not yet hit).
 * Regime 1 = already activated (barrier has been hit).
 * </p>
 *
 * <p>
 * Both regimes evolve under the same 1D PDE operator on the same full spatial grid.
 * The active regime uses the ordinary vanilla PDE and caller-supplied outer boundary conditions.
 * The inactive regime is coupled to the active regime on the whole already-hit region:
 * </p>
 *
 * <ul>
 *   <li>down-in: inactive(x) = active(x) for x &lt;= barrier,</li>
 *   <li>up-in: inactive(x) = active(x) for x &gt;= barrier.</li>
 * </ul>
 *
 * <p>
 * On the continuation-side outer boundary, the inactive regime uses the no-hit asymptotic,
 * i.e. the discounted rebate paid at maturity if the barrier is never hit.
 * </p>
 *
 * <p>
 * The returned surface is the inactive-state surface, since the contract starts in the
 * not-yet-hit regime.
 * </p>
 */
public class FDMThetaMethod1DTwoState implements FDMSolver {

	private static final double EPSILON = 1E-10;

	private final FiniteDifferenceEquityModel model;
	private final BarrierOption product;
	private final SpaceTimeDiscretization spaceTimeDiscretization;
	private final Exercise exercise;
	private final TwoStateActiveBoundaryProvider activeBoundaryProvider;

	public FDMThetaMethod1DTwoState(
			final FiniteDifferenceEquityModel model,
			final BarrierOption product,
			final SpaceTimeDiscretization spaceTimeDiscretization,
			final Exercise exercise,
			final TwoStateActiveBoundaryProvider activeBoundaryProvider) {
		this.model = model;
		this.product = product;
		this.spaceTimeDiscretization = spaceTimeDiscretization;
		this.exercise = exercise;
		this.activeBoundaryProvider = activeBoundaryProvider;
	}

	@Override
	public double[][] getValues(final double time, final DoubleUnaryOperator valueAtMaturity) {

		if(!exercise.isEuropean()) {
			throw new IllegalArgumentException("FDMThetaMethod1DTwoState currently supports only European exercise.");
		}

		if(activeBoundaryProvider == null) {
			throw new IllegalArgumentException("Active boundary provider must not be null.");
		}

		final BarrierType barrierType = product.getBarrierType();
		if(barrierType != BarrierType.DOWN_IN && barrierType != BarrierType.UP_IN) {
			throw new IllegalArgumentException("FDMThetaMethod1DTwoState is only for knock-in barrier options.");
		}

		final double[] xGrid = spaceTimeDiscretization.getSpaceGrid(0).getGrid();
		final int nX = xGrid.length;
		final int n = 2 * nX;

		validateBarrierOnGridNode(xGrid, product.getBarrierValue());

		final double theta = spaceTimeDiscretization.getTheta();
		final int timeLength = spaceTimeDiscretization.getTimeDiscretization().getNumberOfTimeSteps() + 1;
		final int numberOfTimeSteps = spaceTimeDiscretization.getTimeDiscretization().getNumberOfTimeSteps();

		final FiniteDifferenceMatrixBuilder fdBuilder = new FiniteDifferenceMatrixBuilder(xGrid);
		final RealMatrix firstDerivative = fdBuilder.getFirstDerivativeMatrix();
		final RealMatrix secondDerivative = fdBuilder.getSecondDerivativeMatrix();

		final RealMatrix identity1D = MatrixUtils.createRealIdentityMatrix(nX);
		final RealMatrix identity2State = MatrixUtils.createRealIdentityMatrix(n);

		RealMatrix U0 = MatrixUtils.createRealMatrix(nX, 1);
		RealMatrix U1 = MatrixUtils.createRealMatrix(nX, 1);

		for(int i = 0; i < nX; i++) {
			final double x = xGrid[i];
			final double payoff = valueAtMaturity.applyAsDouble(x);

			U1.setEntry(i, 0, payoff);

			if(isAlreadyHitRegion(x, barrierType, product.getBarrierValue())) {
				U0.setEntry(i, 0, payoff);
			}
			else {
				U0.setEntry(i, 0, product.getRebate());
			}
		}

		RealMatrix U = MatrixUtils.createRealMatrix(n, 1);
		copyBlockIntoVector(U0, U, 0);
		copyBlockIntoVector(U1, U, nX);

		final RealMatrix solutionSurface = MatrixUtils.createRealMatrix(nX, timeLength);
		solutionSurface.setColumnMatrix(0, U0);

		for(int m = 0; m < numberOfTimeSteps; m++) {

			final double deltaTau = spaceTimeDiscretization.getTimeDiscretization().getTimeStep(m);

			final double t_m = spaceTimeDiscretization.getTimeDiscretization().getTime(numberOfTimeSteps - m);
			final double t_mp1 = spaceTimeDiscretization.getTimeDiscretization().getTime(numberOfTimeSteps - (m + 1));

			final double tSafe_m = Math.max(t_m, 1E-6);
			final double tSafe_mp1 = Math.max(t_mp1, 1E-6);

			final double r_m = -Math.log(model.getRiskFreeCurve().getDiscountFactor(tSafe_m)) / tSafe_m;
			final double r_mp1 = -Math.log(model.getRiskFreeCurve().getDiscountFactor(tSafe_mp1)) / tSafe_mp1;

			final double[] drift_m = new double[nX];
			final double[] drift_mp1 = new double[nX];
			final double[] variance_m = new double[nX];
			final double[] variance_mp1 = new double[nX];

			for(int i = 0; i < nX; i++) {
				final double x = xGrid[i];

				drift_m[i] = model.getDrift(t_m, x)[0];
				drift_mp1[i] = model.getDrift(t_mp1, x)[0];

				final double[][] factorLoading_m = model.getFactorLoading(t_m, x);
				final double[][] factorLoading_mp1 = model.getFactorLoading(t_mp1, x);

				double localVariance_m = 0.0;
				for(int factor = 0; factor < factorLoading_m[0].length; factor++) {
					final double b = factorLoading_m[0][factor];
					localVariance_m += b * b;
				}

				double localVariance_mp1 = 0.0;
				for(int factor = 0; factor < factorLoading_mp1[0].length; factor++) {
					final double b = factorLoading_mp1[0][factor];
					localVariance_mp1 += b * b;
				}

				variance_m[i] = localVariance_m;
				variance_mp1[i] = localVariance_mp1;
			}

			final RealMatrix driftMatrix_m = new DiagonalMatrix(drift_m);
			final RealMatrix driftMatrix_mp1 = new DiagonalMatrix(drift_mp1);
			final RealMatrix varianceMatrix_m = new DiagonalMatrix(variance_m);
			final RealMatrix varianceMatrix_mp1 = new DiagonalMatrix(variance_mp1);

			final RealMatrix driftTerm_m = driftMatrix_m.scalarMultiply(deltaTau).multiply(firstDerivative);
			final RealMatrix driftTerm_mp1 = driftMatrix_mp1.scalarMultiply(deltaTau).multiply(firstDerivative);

			final RealMatrix diffusionTerm_m = varianceMatrix_m.scalarMultiply(0.5 * deltaTau).multiply(secondDerivative);
			final RealMatrix diffusionTerm_mp1 = varianceMatrix_mp1.scalarMultiply(0.5 * deltaTau).multiply(secondDerivative);

			final RealMatrix forward1D_m =
					identity1D.scalarMultiply(1.0 - r_m * deltaTau)
					.add(driftTerm_m)
					.add(diffusionTerm_m);

			final RealMatrix backward1D_mp1 =
					identity1D.scalarMultiply(1.0 + r_mp1 * deltaTau)
					.subtract(driftTerm_mp1)
					.subtract(diffusionTerm_mp1);

			final RealMatrix forward2State = MatrixUtils.createRealMatrix(n, n);
			final RealMatrix backward2State = MatrixUtils.createRealMatrix(n, n);

			forward2State.setSubMatrix(forward1D_m.getData(), 0, 0);
			forward2State.setSubMatrix(forward1D_m.getData(), nX, nX);

			backward2State.setSubMatrix(backward1D_mp1.getData(), 0, 0);
			backward2State.setSubMatrix(backward1D_mp1.getData(), nX, nX);

			RealMatrix H = backward2State.scalarMultiply(theta).add(identity2State.scalarMultiply(1.0 - theta));
			final RealMatrix A = forward2State.scalarMultiply(1.0 - theta).add(identity2State.scalarMultiply(theta));

			RealMatrix rhs = A.multiply(U);

			final double tau_mp1 = spaceTimeDiscretization.getTimeDiscretization().getTime(m + 1);
			final double currentTime = spaceTimeDiscretization.getTimeDiscretization().getLastTime() - tau_mp1;

			applyOuterBoundaryConditions(H, rhs, currentTime, xGrid, nX, barrierType);

			for(int i = 0; i < nX; i++) {
				if(isAlreadyHitRegion(xGrid[i], barrierType, product.getBarrierValue())) {
					overwriteCouplingRow(H, rhs, i, nX);
				}
			}

			final DecompositionSolver solver = new LUDecomposition(H).getSolver();
			U = solver.solve(rhs);

			for(int i = 0; i < nX; i++) {
				if(isAlreadyHitRegion(xGrid[i], barrierType, product.getBarrierValue())) {
					U.setEntry(i, 0, U.getEntry(nX + i, 0));
				}
			}

			U0 = U.getSubMatrix(0, nX - 1, 0, 0);
			U1 = U.getSubMatrix(nX, 2 * nX - 1, 0, 0);

			solutionSurface.setColumnMatrix(m + 1, U0);
		}

		return solutionSurface.getData();
	}

	@Override
	public double[] getValue(
			final double evaluationTime,
			final double time,
			final DoubleUnaryOperator valueAtMaturity) {

		final RealMatrix values = new Array2DRowRealMatrix(getValues(time, valueAtMaturity));
		final double tau = time - evaluationTime;
		final int timeIndex = spaceTimeDiscretization.getTimeDiscretization().getTimeIndexNearestLessOrEqual(tau);

		return values.getColumn(timeIndex);
	}

	private void applyOuterBoundaryConditions(
			final RealMatrix H,
			final RealMatrix rhs,
			final double currentTime,
			final double[] xGrid,
			final int nX,
			final BarrierType barrierType) {

		final double lowerActiveValue = activeBoundaryProvider.getLowerBoundaryValue(currentTime, xGrid[0]);
		final double upperActiveValue = activeBoundaryProvider.getUpperBoundaryValue(currentTime, xGrid[nX - 1]);

		overwriteDirichletRow(H, rhs, nX, lowerActiveValue);
		overwriteDirichletRow(H, rhs, nX + nX - 1, upperActiveValue);

		final double discountedNoHitValue = getDiscountedNoHitValue(currentTime);

		switch(barrierType) {
		case DOWN_IN:
			overwriteCouplingRow(H, rhs, 0, nX);
			overwriteDirichletRow(H, rhs, nX - 1, discountedNoHitValue);
			break;

		case UP_IN:
			overwriteDirichletRow(H, rhs, 0, discountedNoHitValue);
			overwriteCouplingRow(H, rhs, nX - 1, nX);
			break;

		default:
			throw new IllegalArgumentException("Unsupported barrier type: " + barrierType);
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

		final double discountFactorAtCurrentTime = model.getRiskFreeCurve().getDiscountFactor(t);
		final double discountFactorAtMaturity = model.getRiskFreeCurve().getDiscountFactor(maturity);

		return product.getRebate() * discountFactorAtMaturity / discountFactorAtCurrentTime;
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
			final int nX) {

		for(int col = 0; col < H.getColumnDimension(); col++) {
			H.setEntry(inactiveIndex, col, 0.0);
		}

		H.setEntry(inactiveIndex, inactiveIndex, 1.0);
		H.setEntry(inactiveIndex, nX + inactiveIndex, -1.0);
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
				"Barrier must coincide with a 1D grid node for direct two-state knock-in pricing.");
	}

	private boolean isAlreadyHitRegion(
			final double x,
			final BarrierType barrierType,
			final double barrier) {

		switch(barrierType) {
		case DOWN_IN:
			return x <= barrier;
		case UP_IN:
			return x >= barrier;
		default:
			throw new IllegalArgumentException("Unsupported barrier type: " + barrierType);
		}
	}
}