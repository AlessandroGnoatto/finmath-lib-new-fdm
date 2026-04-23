package net.finmath.finitedifference.assetderivativevaluation.boundaries;

import org.apache.commons.math3.distribution.NormalDistribution;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMMultiAssetBlackScholesModel;
import net.finmath.finitedifference.assetderivativevaluation.products.BasketOption;
import net.finmath.finitedifference.assetderivativevaluation.products.FiniteDifferenceProduct;
import net.finmath.finitedifference.boundaries.BoundaryCondition;
import net.finmath.finitedifference.boundaries.StandardBoundaryCondition;
import net.finmath.marketdata.model.curves.DiscountCurve;
import net.finmath.modelling.products.CallOrPut;

/**
 * Boundary conditions for {@link BasketOption} under
 * {@link MultiAssetBlackScholesModel}.
 *
 * <p>
 * The product payoff is
 * </p>
 *
 * <p>
 * <i>
 * \left( w_1 S_1(T) + w_2 S_2(T) - K \right)^{+}
 * </i>
 * </p>
 *
 * <p>
 * for a call, and
 * </p>
 *
 * <p>
 * <i>
 * \left( K - w_1 S_1(T) - w_2 S_2(T) \right)^{+}
 * </i>
 * </p>
 *
 * <p>
 * for a put, with non-negative weights.
 * </p>
 *
 * <p>
 * Boundary design:
 * </p>
 * <ul>
 *   <li>on {@code S1 = 0}, the payoff reduces exactly to a one-dimensional Black-Scholes
 *       option on {@code S2},</li>
 *   <li>on {@code S2 = 0}, the payoff reduces exactly to a one-dimensional Black-Scholes
 *       option on {@code S1},</li>
 *   <li>on the upper faces, asymptotically correct Dirichlet values are used:
 *       for calls the discounted linear intrinsic approximation, for puts zero.</li>
 * </ul>
 *
 * <p>
 * The returned boundary-condition array follows the library convention:
 * index {@code 0} corresponds to the first state variable and index {@code 1}
 * to the second one.
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public class BasketOptionMultiAssetBlackScholesModelBoundary implements FiniteDifferenceBoundary {

	private static final NormalDistribution NORMAL_DISTRIBUTION = new NormalDistribution();
	private static final double TIME_FLOOR = 1E-10;
	private static final double WEIGHT_TOLERANCE = 1E-14;

	private final FDMMultiAssetBlackScholesModel model;

	public BasketOptionMultiAssetBlackScholesModelBoundary(final FDMMultiAssetBlackScholesModel model) {
		if(model == null) {
			throw new IllegalArgumentException("model must not be null.");
		}
		if(model.getInitialValue() == null || model.getInitialValue().length != 2) {
			throw new IllegalArgumentException(
					"BasketOptionMultiAssetBlackScholesModelBoundary requires a two-dimensional model.");
		}
		this.model = model;
	}

	@Override
	public BoundaryCondition[] getBoundaryConditionsAtLowerBoundary(
			final FiniteDifferenceProduct product,
			final double time,
			final double... stateVariables) {

		final BasketOption basketOption = validateAndCastProduct(product);
		validateStateVariables(stateVariables);

		final double maturity = basketOption.getMaturity();
		final double strike = basketOption.getStrike();
		final double[] weights = basketOption.getWeights();

		final double tau = Math.max(maturity - time, 0.0);
		final double s1 = stateVariables[0];
		final double s2 = stateVariables[1];

		final BoundaryCondition[] conditions = new BoundaryCondition[] {
				StandardBoundaryCondition.none(),
				StandardBoundaryCondition.none()
		};

		if(isAtLowerBoundary(0, s1)) {
			conditions[0] = StandardBoundaryCondition.dirichlet(
					getValueOnS1EqualsZeroFace(
							basketOption.getCallOrPut(),
							weights[1],
							s2,
							strike,
							tau
					)
			);
		}

		if(isAtLowerBoundary(1, s2)) {
			conditions[1] = StandardBoundaryCondition.dirichlet(
					getValueOnS2EqualsZeroFace(
							basketOption.getCallOrPut(),
							weights[0],
							s1,
							strike,
							tau
					)
			);
		}

		return conditions;
	}

	@Override
	public BoundaryCondition[] getBoundaryConditionsAtUpperBoundary(
			final FiniteDifferenceProduct product,
			final double time,
			final double... stateVariables) {

		final BasketOption basketOption = validateAndCastProduct(product);
		validateStateVariables(stateVariables);

		final double maturity = basketOption.getMaturity();
		final double strike = basketOption.getStrike();
		final double[] weights = basketOption.getWeights();

		final double tau = Math.max(maturity - time, 0.0);
		final double s1 = stateVariables[0];
		final double s2 = stateVariables[1];

		final BoundaryCondition[] conditions = new BoundaryCondition[] {
				StandardBoundaryCondition.none(),
				StandardBoundaryCondition.none()
		};

		final double upperFaceValue = getUpperFaceAsymptoticValue(
				basketOption.getCallOrPut(),
				weights[0],
				weights[1],
				s1,
				s2,
				strike,
				tau
		);

		if(isAtUpperBoundary(0, s1)) {
			conditions[0] = StandardBoundaryCondition.dirichlet(upperFaceValue);
		}

		if(isAtUpperBoundary(1, s2)) {
			conditions[1] = StandardBoundaryCondition.dirichlet(upperFaceValue);
		}

		return conditions;
	}

	private BasketOption validateAndCastProduct(final FiniteDifferenceProduct product) {
		if(!(product instanceof BasketOption)) {
			throw new IllegalArgumentException(
					"BasketOptionMultiAssetBlackScholesModelBoundary requires a BasketOption.");
		}

		final BasketOption basketOption = (BasketOption) product;

		if(basketOption.getWeights().length != 2) {
			throw new IllegalArgumentException(
					"BasketOptionMultiAssetBlackScholesModelBoundary currently supports only two assets.");
		}

		if(!basketOption.getExercise().isEuropean()) {
			throw new IllegalArgumentException(
					"BasketOptionMultiAssetBlackScholesModelBoundary currently supports only European exercise.");
		}

		return basketOption;
	}

	private void validateStateVariables(final double[] stateVariables) {
		if(stateVariables == null || stateVariables.length != 2) {
			throw new IllegalArgumentException("Two state variables are required.");
		}
	}

	private boolean isAtLowerBoundary(final int dimension, final double value) {
		final double[] grid = model.getSpaceTimeDiscretization().getSpaceGrid(dimension).getGrid();
		return Math.abs(value - grid[0]) <= 1E-12;
	}

	private boolean isAtUpperBoundary(final int dimension, final double value) {
		final double[] grid = model.getSpaceTimeDiscretization().getSpaceGrid(dimension).getGrid();
		return Math.abs(value - grid[grid.length - 1]) <= 1E-12;
	}

	private double getValueOnS1EqualsZeroFace(
			final CallOrPut callOrPut,
			final double weight2,
			final double s2,
			final double strike,
			final double tau) {
		return scaledBlackScholesValue(
				callOrPut,
				weight2,
				s2,
				strike,
				getDividendYieldRate(1, tau),
				model.getVolatilities()[1],
				tau
		);
	}

	private double getValueOnS2EqualsZeroFace(
			final CallOrPut callOrPut,
			final double weight1,
			final double s1,
			final double strike,
			final double tau) {
		return scaledBlackScholesValue(
				callOrPut,
				weight1,
				s1,
				strike,
				getDividendYieldRate(0, tau),
				model.getVolatilities()[0],
				tau
		);
	}

	private double getUpperFaceAsymptoticValue(
			final CallOrPut callOrPut,
			final double weight1,
			final double weight2,
			final double s1,
			final double s2,
			final double strike,
			final double tau) {

		if(callOrPut == CallOrPut.PUT) {
			return 0.0;
		}

		final double discountedBasketForward =
				weight1 * s1 * getDividendDiscountFactor(0, tau)
				+ weight2 * s2 * getDividendDiscountFactor(1, tau);

		final double discountedStrike = strike * getRiskFreeDiscountFactor(tau);

		return Math.max(discountedBasketForward - discountedStrike, 0.0);
	}

	private double scaledBlackScholesValue(
			final CallOrPut callOrPut,
			final double weight,
			final double spot,
			final double strike,
			final double dividendYield,
			final double volatility,
			final double tau) {

		if(weight <= WEIGHT_TOLERANCE) {
			if(callOrPut == CallOrPut.CALL) {
				return 0.0;
			}
			return strike * getRiskFreeDiscountFactor(tau);
		}

		if(tau <= 0.0) {
			if(callOrPut == CallOrPut.CALL) {
				return Math.max(weight * spot - strike, 0.0);
			}
			return Math.max(strike - weight * spot, 0.0);
		}

		return weight * blackScholesValue(
				callOrPut,
				spot,
				strike / weight,
				getRiskFreeRate(tau),
				dividendYield,
				volatility,
				tau
		);
	}

	private double blackScholesValue(
			final CallOrPut callOrPut,
			final double spot,
			final double strike,
			final double riskFreeRate,
			final double dividendYield,
			final double volatility,
			final double tau) {

		if(spot <= 0.0) {
			if(callOrPut == CallOrPut.CALL) {
				return 0.0;
			}
			return strike * Math.exp(-riskFreeRate * tau);
		}

		if(strike <= 0.0) {
			if(callOrPut == CallOrPut.CALL) {
				return spot * Math.exp(-dividendYield * tau);
			}
			return 0.0;
		}

		if(volatility <= 0.0) {
			final double deterministic =
					spot * Math.exp(-dividendYield * tau)
					- strike * Math.exp(-riskFreeRate * tau);

			if(callOrPut == CallOrPut.CALL) {
				return Math.max(deterministic, 0.0);
			}
			return Math.max(-deterministic, 0.0);
		}

		final double sqrtTau = Math.sqrt(tau);
		final double sigmaSqrtTau = volatility * sqrtTau;

		final double d1 =
				(
						Math.log(spot / strike)
						+ (riskFreeRate - dividendYield + 0.5 * volatility * volatility) * tau
				) / sigmaSqrtTau;
		final double d2 = d1 - sigmaSqrtTau;

		final double discountedSpot = spot * Math.exp(-dividendYield * tau);
		final double discountedStrike = strike * Math.exp(-riskFreeRate * tau);

		if(callOrPut == CallOrPut.CALL) {
			return discountedSpot * NORMAL_DISTRIBUTION.cumulativeProbability(d1)
					- discountedStrike * NORMAL_DISTRIBUTION.cumulativeProbability(d2);
		}
		return discountedStrike * NORMAL_DISTRIBUTION.cumulativeProbability(-d2)
				- discountedSpot * NORMAL_DISTRIBUTION.cumulativeProbability(-d1);
	}

	private double getRiskFreeRate(final double tau) {
		final double safeTau = Math.max(tau, TIME_FLOOR);
		final DiscountCurve curve = model.getRiskFreeCurve();
		return -Math.log(curve.getDiscountFactor(safeTau)) / safeTau;
	}

	private double getDividendYieldRate(final int assetIndex, final double tau) {
		final double safeTau = Math.max(tau, TIME_FLOOR);
		final DiscountCurve curve = model.getDividendYieldCurves()[assetIndex];
		return -Math.log(curve.getDiscountFactor(safeTau)) / safeTau;
	}

	private double getRiskFreeDiscountFactor(final double tau) {
		if(tau <= 0.0) {
			return 1.0;
		}
		return model.getRiskFreeCurve().getDiscountFactor(tau);
	}

	private double getDividendDiscountFactor(final int assetIndex, final double tau) {
		if(tau <= 0.0) {
			return 1.0;
		}
		return model.getDividendYieldCurves()[assetIndex].getDiscountFactor(tau);
	}
}