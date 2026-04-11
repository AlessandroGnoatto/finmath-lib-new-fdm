package net.finmath.finitedifference.assetderivativevaluation.products;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMBlackScholesModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMBachelierModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMCevModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.solvers.FDMSolver;
import net.finmath.finitedifference.solvers.FDMSolverFactory;
import net.finmath.modelling.EuropeanExercise;
import net.finmath.modelling.Exercise;
import net.finmath.modelling.products.CallOrPut;

/**
 * European digital option supporting cash-or-nothing and asset-or-nothing payoffs.
 *
 * <p>
 * This implementation uses cell-averaged terminal values on one-dimensional
 * grids in order to reduce the grid bias caused by the payoff discontinuity
 * at the strike.
 * </p>
 *
 * <p>
 * For a one-dimensional model, the terminal layer is built directly on the
 * spatial grid using cell averages over the local dual cell around each node.
 * </p>
 *
 * <p>
 * This class currently supports European exercise only.
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public class DigitalOption implements FiniteDifferenceProduct {

	public enum DigitalPayoffType {
		CASH_OR_NOTHING,
		ASSET_OR_NOTHING
	}

	private final String underlyingName;
	private final double maturity;
	private final double strike;
	private final CallOrPut callOrPutSign;
	private final DigitalPayoffType digitalPayoffType;
	private final double cashPayoff;
	private final Exercise exercise;

	/**
	 * Creates a digital option.
	 *
	 * @param underlyingName Name of the underlying.
	 * @param maturity Option maturity.
	 * @param strike Option strike.
	 * @param callOrPutSign Call or put flag.
	 * @param digitalPayoffType Cash-or-nothing or asset-or-nothing.
	 * @param cashPayoff Cash payoff amount for cash-or-nothing.
	 */
	public DigitalOption(
			final String underlyingName,
			final double maturity,
			final double strike,
			final CallOrPut callOrPutSign,
			final DigitalPayoffType digitalPayoffType,
			final double cashPayoff) {

		if(callOrPutSign == null) {
			throw new IllegalArgumentException("Option type must not be null.");
		}
		if(digitalPayoffType == null) {
			throw new IllegalArgumentException("Digital payoff type must not be null.");
		}
		if(digitalPayoffType == DigitalPayoffType.CASH_OR_NOTHING && cashPayoff < 0.0) {
			throw new IllegalArgumentException("Cash payoff must be non-negative.");
		}

		this.underlyingName = underlyingName;
		this.maturity = maturity;
		this.strike = strike;
		this.callOrPutSign = callOrPutSign;
		this.digitalPayoffType = digitalPayoffType;
		this.cashPayoff = digitalPayoffType == DigitalPayoffType.CASH_OR_NOTHING ? cashPayoff : Double.NaN;
		this.exercise = new EuropeanExercise(maturity);
	}

	/**
	 * Creates a digital option.
	 *
	 * @param maturity Option maturity.
	 * @param strike Option strike.
	 * @param callOrPutSign Call or put flag.
	 * @param digitalPayoffType Cash-or-nothing or asset-or-nothing.
	 * @param cashPayoff Cash payoff amount for cash-or-nothing.
	 */
	public DigitalOption(
			final double maturity,
			final double strike,
			final CallOrPut callOrPutSign,
			final DigitalPayoffType digitalPayoffType,
			final double cashPayoff) {
		this(null, maturity, strike, callOrPutSign, digitalPayoffType, cashPayoff);
	}

	/**
	 * Creates a digital option.
	 *
	 * @param underlyingName Name of the underlying.
	 * @param maturity Option maturity.
	 * @param strike Option strike.
	 * @param callOrPutSign +1 for call, -1 for put.
	 * @param digitalPayoffType Cash-or-nothing or asset-or-nothing.
	 * @param cashPayoff Cash payoff amount for cash-or-nothing.
	 */
	public DigitalOption(
			final String underlyingName,
			final double maturity,
			final double strike,
			final double callOrPutSign,
			final DigitalPayoffType digitalPayoffType,
			final double cashPayoff) {
		this(underlyingName, maturity, strike, mapCallOrPut(callOrPutSign), digitalPayoffType, cashPayoff);
	}

	@Override
	public double[] getValue(final double evaluationTime, final FiniteDifferenceEquityModel model) {

		final FDMSolver solver = FDMSolverFactory.createSolver(model, this, exercise);

		if(isOneDimensionalModel(model)) {
			final double[] terminalValues = buildCellAveragedTerminalValues(model);
			return solver.getValue(evaluationTime, maturity, terminalValues);
		}

		return solver.getValue(evaluationTime, maturity, this::payoff);
	}

	@Override
	public double[][] getValues(final FiniteDifferenceEquityModel model) {

		final FDMSolver solver = FDMSolverFactory.createSolver(model, this, exercise);

		if(isOneDimensionalModel(model)) {
			final double[] terminalValues = buildCellAveragedTerminalValues(model);
			return solver.getValues(maturity, terminalValues);
		}

		return solver.getValues(maturity, this::payoff);
	}

	/**
	 * Pointwise payoff. Used as fallback for models where cell-averaged terminal
	 * initialization is not yet available.
	 *
	 * @param assetValue Asset value.
	 * @return Payoff at maturity.
	 */
	private double payoff(final double assetValue) {

		final boolean inTheMoney =
				callOrPutSign == CallOrPut.CALL
						? assetValue > strike
						: assetValue < strike;

		if(!inTheMoney) {
			return 0.0;
		}

		switch(digitalPayoffType) {
		case CASH_OR_NOTHING:
			return cashPayoff;
		case ASSET_OR_NOTHING:
			return assetValue;
		default:
			throw new IllegalStateException("Unsupported digital payoff type.");
		}
	}

	/**
	 * Builds the cell-averaged terminal values on the one-dimensional spot grid.
	 *
	 * @param model The finite-difference model.
	 * @return Terminal values on the spot grid.
	 */
	private double[] buildCellAveragedTerminalValues(final FiniteDifferenceEquityModel model) {

		final double[] sGrid = getSpotGrid(model);
		final double[] terminalValues = new double[sGrid.length];

		for(int i = 0; i < sGrid.length; i++) {
			final double leftEdge = getLeftDualCellEdge(sGrid, i);
			final double rightEdge = getRightDualCellEdge(sGrid, i);
			terminalValues[i] = cellAveragedPayoff(leftEdge, rightEdge);
		}

		return terminalValues;
	}

	/**
	 * Returns the payoff averaged over a cell.
	 *
	 * @param leftEdge Left cell edge.
	 * @param rightEdge Right cell edge.
	 * @return Cell-averaged payoff.
	 */
	private double cellAveragedPayoff(final double leftEdge, final double rightEdge) {

		if(!(leftEdge < rightEdge)) {
			throw new IllegalArgumentException("Require leftEdge < rightEdge.");
		}

		switch(digitalPayoffType) {
		case CASH_OR_NOTHING:
			return cellAveragedCashDigital(leftEdge, rightEdge);
		case ASSET_OR_NOTHING:
			return cellAveragedAssetDigital(leftEdge, rightEdge);
		default:
			throw new IllegalStateException("Unsupported digital payoff type.");
		}
	}

	/**
	 * Cell average for a cash-or-nothing digital payoff.
	 *
	 * @param leftEdge Left cell edge.
	 * @param rightEdge Right cell edge.
	 * @return Cell-averaged payoff.
	 */
	private double cellAveragedCashDigital(final double leftEdge, final double rightEdge) {

		final double cellLength = rightEdge - leftEdge;

		if(callOrPutSign == CallOrPut.CALL) {
			if(rightEdge <= strike) {
				return 0.0;
			}
			if(leftEdge >= strike) {
				return cashPayoff;
			}
			return cashPayoff * (rightEdge - strike) / cellLength;
		}
		else {
			if(rightEdge <= strike) {
				return cashPayoff;
			}
			if(leftEdge >= strike) {
				return 0.0;
			}
			return cashPayoff * (strike - leftEdge) / cellLength;
		}
	}

	/**
	 * Cell average for an asset-or-nothing digital payoff.
	 *
	 * @param leftEdge Left cell edge.
	 * @param rightEdge Right cell edge.
	 * @return Cell-averaged payoff.
	 */
	private double cellAveragedAssetDigital(final double leftEdge, final double rightEdge) {

		final double cellLength = rightEdge - leftEdge;

		if(callOrPutSign == CallOrPut.CALL) {
			if(rightEdge <= strike) {
				return 0.0;
			}
			if(leftEdge >= strike) {
				return 0.5 * (leftEdge + rightEdge);
			}
			return (rightEdge * rightEdge - strike * strike) / (2.0 * cellLength);
		}
		else {
			if(rightEdge <= strike) {
				return 0.5 * (leftEdge + rightEdge);
			}
			if(leftEdge >= strike) {
				return 0.0;
			}
			return (strike * strike - leftEdge * leftEdge) / (2.0 * cellLength);
		}
	}

	/**
	 * Returns the left edge of the dual cell around node i.
	 *
	 * @param grid Spot grid.
	 * @param i Node index.
	 * @return Left dual-cell edge.
	 */
	private double getLeftDualCellEdge(final double[] grid, final int i) {
		if(i == 0) {
			return grid[0];
		}
		return 0.5 * (grid[i - 1] + grid[i]);
	}

	/**
	 * Returns the right edge of the dual cell around node i.
	 *
	 * @param grid Spot grid.
	 * @param i Node index.
	 * @return Right dual-cell edge.
	 */
	private double getRightDualCellEdge(final double[] grid, final int i) {
		if(i == grid.length - 1) {
			return grid[grid.length - 1];
		}
		return 0.5 * (grid[i] + grid[i + 1]);
	}

	/**
	 * Returns the one-dimensional spot grid from the model.
	 *
	 * @param model The model.
	 * @return The spot grid.
	 */
	private double[] getSpotGrid(final FiniteDifferenceEquityModel model) {
		final Grid grid = model.getSpaceTimeDiscretization().getSpaceGrid(0);
		return grid.getGrid();
	}

	/**
	 * Checks whether the model is one-dimensional.
	 *
	 * @param model The model.
	 * @return True if the model is one-dimensional.
	 */
	private boolean isOneDimensionalModel(final FiniteDifferenceEquityModel model) {
		return model instanceof FDMBlackScholesModel
				|| model instanceof FDMCevModel
				|| model instanceof FDMBachelierModel;
	}

	private static CallOrPut mapCallOrPut(final double callOrPutSign) {
		if(callOrPutSign == 1.0) {
			return CallOrPut.CALL;
		}
		if(callOrPutSign == -1.0) {
			return CallOrPut.PUT;
		}
		throw new IllegalArgumentException("Unknown option type.");
	}

	public String getUnderlyingName() {
		return underlyingName;
	}

	public double getMaturity() {
		return maturity;
	}

	public double getStrike() {
		return strike;
	}

	public CallOrPut getCallOrPut() {
		return callOrPutSign;
	}

	public DigitalPayoffType getDigitalPayoffType() {
		return digitalPayoffType;
	}

	public double getCashPayoff() {
		return cashPayoff;
	}

	public Exercise getExercise() {
		return exercise;
	}
}