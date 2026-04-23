package net.finmath.functions2;

import org.apache.commons.math3.distribution.NormalDistribution;

/**
 * Additional analytic pricing formulas not yet contained in the standard
 * analytic-formula helper classes.
 *
 * <p>
 * This utility class currently provides the Margrabe formula for a European
 * exchange option under the two-asset Black-Scholes model with continuous
 * dividend yields.
 * </p>
 *
 * <p>
 * Under the risk-neutral measure, let the asset dynamics be
 * </p>
 *
 * <p>
 * <i>
 * dS_i(t) = (r - q_i) S_i(t) dt + sigma_i S_i(t) dW_i(t),
 * </i>
 * </p>
 *
 * <p>
 * for {@code i = 1,2}, with
 * </p>
 *
 * <p>
 * <i>
 * d&lt;W_1, W_2&gt;_t = rho dt.
 * </i>
 * </p>
 *
 * <p>
 * The price of the exchange option with payoff
 * </p>
 *
 * <p>
 * <i>
 * (S_1(T) - S_2(T))^{+}
 * </i>
 * </p>
 *
 * <p>
 * is given by the Margrabe formula
 * </p>
 *
 * <p>
 * <i>
 * V_0 = S_1(0) e^{-q_1 T} N(d_1) - S_2(0) e^{-q_2 T} N(d_2),
 * </i>
 * </p>
 *
 * <p>
 * where
 * </p>
 *
 * <p>
 * <i>
 * sigma_M = sqrt(sigma_1^2 + sigma_2^2 - 2 rho sigma_1 sigma_2),
 * </i>
 * </p>
 *
 * <p>
 * <i>
 * d_1 = \frac{\ln(S_1(0)/S_2(0)) + (q_2 - q_1 + \frac{1}{2} sigma_M^2) T}
 * {sigma_M \sqrt{T}},
 * </i>
 * </p>
 *
 * <p>
 * and
 * </p>
 *
 * <p>
 * <i>
 * d_2 = d_1 - sigma_M \sqrt{T}.
 * </i>
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public class AnalyticFormulas2 {

	private static final NormalDistribution NORMAL_DISTRIBUTION = new NormalDistribution();

	private AnalyticFormulas2() {
		// Utility class
	}

	/**
	 * Returns the Margrabe price of a European exchange option with payoff
	 * {@code max(S1(T) - S2(T), 0)}.
	 *
	 * @param initialValueFirstAsset Initial value of the first asset.
	 * @param initialValueSecondAsset Initial value of the second asset.
	 * @param dividendYieldFirstAsset Continuous dividend yield of the first asset.
	 * @param dividendYieldSecondAsset Continuous dividend yield of the second asset.
	 * @param volatilityFirstAsset Volatility of the first asset.
	 * @param volatilitySecondAsset Volatility of the second asset.
	 * @param correlation Instantaneous correlation between the two assets.
	 * @param maturity Option maturity.
	 * @return The Margrabe price.
	 */
	public static double margrabeExchangeOptionValue(
			final double initialValueFirstAsset,
			final double initialValueSecondAsset,
			final double dividendYieldFirstAsset,
			final double dividendYieldSecondAsset,
			final double volatilityFirstAsset,
			final double volatilitySecondAsset,
			final double correlation,
			final double maturity) {

		if(initialValueFirstAsset <= 0.0 || initialValueSecondAsset <= 0.0) {
			throw new IllegalArgumentException("Initial asset values must be strictly positive.");
		}
		if(volatilityFirstAsset < 0.0 || volatilitySecondAsset < 0.0) {
			throw new IllegalArgumentException("Volatilities must be non-negative.");
		}
		if(correlation < -1.0 || correlation > 1.0) {
			throw new IllegalArgumentException("Correlation must lie in [-1,1].");
		}
		if(maturity < 0.0) {
			throw new IllegalArgumentException("Maturity must be non-negative.");
		}

		if(maturity == 0.0) {
			return Math.max(initialValueFirstAsset - initialValueSecondAsset, 0.0);
		}

		final double effectiveVolatilitySquared =
				volatilityFirstAsset * volatilityFirstAsset
				+ volatilitySecondAsset * volatilitySecondAsset
				- 2.0 * correlation * volatilityFirstAsset * volatilitySecondAsset;

		final double effectiveVolatility = Math.sqrt(Math.max(effectiveVolatilitySquared, 0.0));
		final double sqrtMaturity = Math.sqrt(maturity);

		if(effectiveVolatility == 0.0) {
			final double deterministicForwardDifference =
					initialValueFirstAsset * Math.exp(-dividendYieldFirstAsset * maturity)
					- initialValueSecondAsset * Math.exp(-dividendYieldSecondAsset * maturity);

			return Math.max(deterministicForwardDifference, 0.0);
		}

		final double d1 =
				(
						Math.log(initialValueFirstAsset / initialValueSecondAsset)
						+ (dividendYieldSecondAsset - dividendYieldFirstAsset
								+ 0.5 * effectiveVolatilitySquared) * maturity
				)
				/ (effectiveVolatility * sqrtMaturity);

		final double d2 = d1 - effectiveVolatility * sqrtMaturity;

		return initialValueFirstAsset * Math.exp(-dividendYieldFirstAsset * maturity) * NORMAL_DISTRIBUTION.cumulativeProbability(d1)
				- initialValueSecondAsset * Math.exp(-dividendYieldSecondAsset * maturity) * NORMAL_DISTRIBUTION.cumulativeProbability(d2);
	}
}