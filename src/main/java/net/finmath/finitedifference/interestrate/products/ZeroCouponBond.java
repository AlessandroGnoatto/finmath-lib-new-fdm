package net.finmath.finitedifference.interestrate.products;

import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.interestrate.models.FiniteDifferenceInterestRateModel;
import net.finmath.finitedifference.solvers.FDMThetaMethod1D;
import net.finmath.modelling.EuropeanExercise;

/**
 * Finite-difference valuation of a zero-coupon bond.
 *
 * <p>
 * The product pays the notional at maturity and nothing before maturity.
 * In particular, its event representation is
 * </p>
 *
 * <p>
 * <i>
 * V(T^{-},x) = V(T^{+},x) + N,
 * </i>
 * </p>
 *
 * <p>
 * where {@code N} denotes the notional and {@code T} the maturity.
 * </p>
 *
 * <p>
 * Although one-factor Hull-White models provide discount-bond values in closed
 * form through the model method {@code getDiscountBond(t,T,...)}, this class is
 * still useful as:
 * </p>
 * <ul>
 *   <li>a first validation product for the rates finite-difference framework,</li>
 *   <li>a first use case of the event-condition mechanism,</li>
 *   <li>a building block for coupon bonds and more general interest-rate products.</li>
 * </ul>
 *
 * <p>
 * The current implementation uses the one-dimensional theta-method solver and
 * therefore requires a one-dimensional space discretization.
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public class ZeroCouponBond implements FiniteDifferenceInterestRateProduct {

	private static final double EVENT_TIME_TOLERANCE = 1E-12;

	private final double maturity;
	private final double notional;

	/**
	 * Creates a zero-coupon bond with unit notional.
	 *
	 * @param maturity The maturity.
	 */
	public ZeroCouponBond(final double maturity) {
		this(maturity, 1.0);
	}

	/**
	 * Creates a zero-coupon bond.
	 *
	 * @param maturity The maturity.
	 * @param notional The notional paid at maturity.
	 */
	public ZeroCouponBond(final double maturity, final double notional) {
		if(maturity < 0.0) {
			throw new IllegalArgumentException("maturity must be non-negative.");
		}
		if(notional < 0.0) {
			throw new IllegalArgumentException("notional must be non-negative.");
		}

		this.maturity = maturity;
		this.notional = notional;
	}

	@Override
	public double[] getValue(
			final double evaluationTime,
			final FiniteDifferenceInterestRateModel model) {

		validateModel(model);

		final FDMThetaMethod1D solver = new FDMThetaMethod1D(
				model,
				this,
				model.getSpaceTimeDiscretization(),
				new EuropeanExercise(maturity)
		);

		return solver.getValue(
				evaluationTime,
				maturity,
				buildZeroTerminalValues(model)
		);
	}

	@Override
	public double[][] getValues(final FiniteDifferenceInterestRateModel model) {

		validateModel(model);

		final FDMThetaMethod1D solver = new FDMThetaMethod1D(
				model,
				this,
				model.getSpaceTimeDiscretization(),
				new EuropeanExercise(maturity)
		);

		return solver.getValues(
				maturity,
				buildZeroTerminalValues(model)
		);
	}

	@Override
	public double[] getEventTimes() {
		return new double[] { maturity };
	}

	@Override
	public double[] applyEventCondition(
			final double time,
			final double[] valuesAfterEvent,
			final FiniteDifferenceInterestRateModel model) {

		if(valuesAfterEvent == null) {
			throw new IllegalArgumentException("valuesAfterEvent must not be null.");
		}

		if(Math.abs(time - maturity) > EVENT_TIME_TOLERANCE) {
			return valuesAfterEvent;
		}

		final double[] valuesBeforeEvent = valuesAfterEvent.clone();
		for(int i = 0; i < valuesBeforeEvent.length; i++) {
			valuesBeforeEvent[i] += notional;
		}

		return valuesBeforeEvent;
	}

	/**
	 * Returns the maturity.
	 *
	 * @return The maturity.
	 */
	public double getMaturity() {
		return maturity;
	}

	/**
	 * Returns the notional.
	 *
	 * @return The notional.
	 */
	public double getNotional() {
		return notional;
	}

	private void validateModel(final FiniteDifferenceInterestRateModel model) {
		if(model == null) {
			throw new IllegalArgumentException("model must not be null.");
		}
		if(model.getSpaceTimeDiscretization().getNumberOfSpaceGrids() != 1) {
			throw new IllegalArgumentException(
					"ZeroCouponBond currently supports only one-dimensional finite-difference interest-rate models."
			);
		}
	}

	private double[] buildZeroTerminalValues(final FiniteDifferenceInterestRateModel model) {
		final SpaceTimeDiscretization discretization = model.getSpaceTimeDiscretization();
		final double[] xGrid = discretization.getSpaceGrid(0).getGrid();
		return new double[xGrid.length];
	}
}