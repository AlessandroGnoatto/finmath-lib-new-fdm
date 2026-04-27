package net.finmath.finitedifference.interestrate.products;

import net.finmath.finitedifference.FiniteDifferenceProduct;
import net.finmath.finitedifference.interestrate.models.FiniteDifferenceInterestRateModel;

/**
 * Interface for products valued by a finite-difference interest-rate model.
 *
 * <p>
 * This interface specializes the generic
 * {@link net.finmath.finitedifference.FiniteDifferenceProduct} to the case of
 * interest-rate finite-difference models.
 * </p>
 *
 * <p>
 * Interest-rate products typically involve event times such as fixing dates,
 * coupon dates, payment dates, exercise dates, call dates, or redemption dates.
 * These are exposed via {@link #getEventTimes()}.
 * </p>
 *
 * <p>
 * Products without intermediate events may return an empty array.
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public interface FiniteDifferenceInterestRateProduct
	extends FiniteDifferenceProduct<FiniteDifferenceInterestRateModel> {

	@Override
	default Class<FiniteDifferenceInterestRateModel> getModelClass() {
		return FiniteDifferenceInterestRateModel.class;
	}

	/**
	 * Returns the event times of the product.
	 *
	 * <p>
	 * Event times are the dates where the backward induction may have to apply a
	 * jump or another event condition, for example because of coupon accrual,
	 * coupon payment, fixing, exercise, callability, or redemption.
	 * </p>
	 *
	 * <p>
	 * Products without intermediate events may return an empty array.
	 * </p>
	 *
	 * @return The event times of the product.
	 */
	default double[] getEventTimes() {
		return new double[0];
	}
}