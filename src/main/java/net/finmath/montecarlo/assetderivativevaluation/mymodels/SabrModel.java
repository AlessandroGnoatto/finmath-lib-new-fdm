package net.finmath.montecarlo.assetderivativevaluation.mymodels;

import java.time.LocalDate;
import java.util.Map;

import net.finmath.marketdata.model.curves.CurveInterpolation.ExtrapolationMethod;
import net.finmath.marketdata.model.curves.CurveInterpolation.InterpolationEntity;
import net.finmath.marketdata.model.curves.CurveInterpolation.InterpolationMethod;
import net.finmath.marketdata.model.curves.DiscountCurve;
import net.finmath.marketdata.model.curves.DiscountCurveInterpolation;
import net.finmath.montecarlo.RandomVariableFactory;
import net.finmath.montecarlo.RandomVariableFromArrayFactory;
import net.finmath.montecarlo.model.AbstractProcessModel;
import net.finmath.montecarlo.process.MonteCarloProcess;
import net.finmath.stochastic.RandomVariable;

public class SabrModel extends AbstractProcessModel {

	private final RandomVariable initialValue;
	private final RandomVariable alpha;
	private final RandomVariable beta;
	private final RandomVariable nu;
	private final RandomVariable rho;
	private final RandomVariable rhoBar;

	private final DiscountCurve discountCurveForForwardRate;
	private final DiscountCurve discountCurveForDividendYield;
	private final DiscountCurve discountCurveForDiscountRate;

	private final RandomVariable riskFreeRate;
	private final RandomVariable dividendYieldRate;
	private final RandomVariable discountRate;

	private final RandomVariableFactory randomVariableFactory;

	private final RandomVariable[] initialStateVector = new RandomVariable[2];

	public SabrModel(
			final RandomVariable initialValue,
			final DiscountCurve discountCurveForForwardRate,
			final DiscountCurve discountCurveForDividendYield,
			final RandomVariable alpha,
			final RandomVariable beta,
			final RandomVariable nu,
			final RandomVariable rho,
			final DiscountCurve discountCurveForDiscountRate,
			final RandomVariableFactory randomVariableFactory) {
		super();

		this.initialValue = initialValue;
		this.discountCurveForForwardRate = discountCurveForForwardRate;
		this.discountCurveForDividendYield = discountCurveForDividendYield;
		this.alpha = alpha;
		this.beta = beta;
		this.nu = nu;
		this.rho = rho;
		this.rhoBar = rho.squared().mult(-1.0).add(1.0).floor(0.0).sqrt();
		this.discountCurveForDiscountRate = discountCurveForDiscountRate;

		this.riskFreeRate = null;
		this.dividendYieldRate = null;
		this.discountRate = null;

		this.randomVariableFactory = randomVariableFactory;
	}

	public SabrModel(
			final RandomVariable initialValue,
			final DiscountCurve discountCurveForForwardRate,
			final RandomVariable alpha,
			final RandomVariable beta,
			final RandomVariable nu,
			final RandomVariable rho,
			final DiscountCurve discountCurveForDiscountRate,
			final RandomVariableFactory randomVariableFactory) {
		this(
				initialValue,
				discountCurveForForwardRate,
				createFlatDiscountCurve("dividendCurve", 0.0),
				alpha,
				beta,
				nu,
				rho,
				discountCurveForDiscountRate,
				randomVariableFactory
		);
	}

	public SabrModel(
			final RandomVariable initialValue,
			final RandomVariable riskFreeRate,
			final RandomVariable dividendYieldRate,
			final RandomVariable alpha,
			final RandomVariable beta,
			final RandomVariable nu,
			final RandomVariable rho,
			final RandomVariable discountRate,
			final RandomVariableFactory randomVariableFactory) {
		super();

		this.initialValue = initialValue;
		this.riskFreeRate = riskFreeRate;
		this.dividendYieldRate = dividendYieldRate;
		this.alpha = alpha;
		this.beta = beta;
		this.nu = nu;
		this.rho = rho;
		this.rhoBar = rho.squared().mult(-1.0).add(1.0).floor(0.0).sqrt();
		this.discountRate = discountRate;

		this.discountCurveForForwardRate = null;
		this.discountCurveForDividendYield = null;
		this.discountCurveForDiscountRate = null;

		this.randomVariableFactory = randomVariableFactory;
	}

	public SabrModel(
			final RandomVariable initialValue,
			final RandomVariable riskFreeRate,
			final RandomVariable alpha,
			final RandomVariable beta,
			final RandomVariable nu,
			final RandomVariable rho,
			final RandomVariable discountRate,
			final RandomVariableFactory randomVariableFactory) {
		this(
				initialValue,
				riskFreeRate,
				randomVariableFactory.createRandomVariable(0.0),
				alpha,
				beta,
				nu,
				rho,
				discountRate,
				randomVariableFactory
		);
	}

	public SabrModel(
			final double initialValue,
			final double riskFreeRate,
			final double dividendYieldRate,
			final double alpha,
			final double beta,
			final double nu,
			final double rho,
			final double discountRate,
			final RandomVariableFactory randomVariableFactory) {
		this(
				randomVariableFactory.createRandomVariable(initialValue),
				randomVariableFactory.createRandomVariable(riskFreeRate),
				randomVariableFactory.createRandomVariable(dividendYieldRate),
				randomVariableFactory.createRandomVariable(alpha),
				randomVariableFactory.createRandomVariable(beta),
				randomVariableFactory.createRandomVariable(nu),
				randomVariableFactory.createRandomVariable(rho),
				randomVariableFactory.createRandomVariable(discountRate),
				randomVariableFactory
		);
	}

	public SabrModel(
			final double initialValue,
			final double riskFreeRate,
			final double dividendYieldRate,
			final double alpha,
			final double beta,
			final double nu,
			final double rho) {
		this(
				initialValue,
				riskFreeRate,
				dividendYieldRate,
				alpha,
				beta,
				nu,
				rho,
				riskFreeRate,
				new RandomVariableFromArrayFactory()
		);
	}

	public SabrModel(
			final double initialValue,
			final double riskFreeRate,
			final double alpha,
			final double beta,
			final double nu,
			final double rho) {
		this(
				initialValue,
				riskFreeRate,
				0.0,
				alpha,
				beta,
				nu,
				rho,
				riskFreeRate,
				new RandomVariableFromArrayFactory()
		);
	}

	@Override
	public RandomVariable[] getInitialState(final MonteCarloProcess process) {
		if(initialStateVector[0] == null) {
			initialStateVector[0] = initialValue.log();
			initialStateVector[1] = alpha.log();
		}
		return initialStateVector;
	}

	@Override
	public RandomVariable[] getDrift(
			final MonteCarloProcess process,
			final int timeIndex,
			final RandomVariable[] realizationAtTimeIndex,
			final RandomVariable[] realizationPredictor) {

		final RandomVariable x1 = realizationAtTimeIndex[0];
		final RandomVariable x2 = realizationAtTimeIndex[1];

		final RandomVariable localAlpha = x2.exp();

		final RandomVariable riskFreeRateAtTimeStep = getRiskFreeRateAtTimeStep(process, timeIndex);
		final RandomVariable dividendYieldRateAtTimeStep = getDividendYieldRateAtTimeStep(process, timeIndex);

		final RandomVariable spotToBetaMinusOne = x1.mult(beta.sub(1.0)).exp();
		final RandomVariable sigmaX1 = localAlpha.mult(spotToBetaMinusOne);

		final RandomVariable driftX1 =
				riskFreeRateAtTimeStep.sub(dividendYieldRateAtTimeStep).sub(sigmaX1.squared().div(2.0));

		final RandomVariable driftX2 = nu.squared().div(-2.0);

		return new RandomVariable[] { driftX1, driftX2 };
	}

	@Override
	public RandomVariable[] getFactorLoading(
			final MonteCarloProcess process,
			final int timeIndex,
			final int component,
			final RandomVariable[] realizationAtTimeIndex) {

		final RandomVariable x1 = realizationAtTimeIndex[0];
		final RandomVariable x2 = realizationAtTimeIndex[1];

		final RandomVariable localAlpha = x2.exp();
		final RandomVariable spotToBetaMinusOne = x1.mult(beta.sub(1.0)).exp();
		final RandomVariable sigmaX1 = localAlpha.mult(spotToBetaMinusOne);

		final RandomVariable[] factorLoadings = new RandomVariable[2];

		if(component == 0) {
			factorLoadings[0] = sigmaX1;
			factorLoadings[1] = getRandomVariableForConstant(0.0);
		}
		else if(component == 1) {
			factorLoadings[0] = nu.mult(rho);
			factorLoadings[1] = nu.mult(rhoBar);
		}
		else {
			throw new UnsupportedOperationException("Component " + component + " does not exist.");
		}

		return factorLoadings;
	}

	@Override
	public RandomVariable applyStateSpaceTransform(
			final MonteCarloProcess process,
			final int timeIndex,
			final int componentIndex,
			final RandomVariable randomVariable) {

		if(componentIndex == 0 || componentIndex == 1) {
			return randomVariable.exp();
		}
		else {
			throw new UnsupportedOperationException("Component " + componentIndex + " does not exist.");
		}
	}

	@Override
	public RandomVariable applyStateSpaceTransformInverse(
			final MonteCarloProcess process,
			final int timeIndex,
			final int componentIndex,
			final RandomVariable randomVariable) {

		if(componentIndex == 0 || componentIndex == 1) {
			return randomVariable.log();
		}
		else {
			throw new UnsupportedOperationException("Component " + componentIndex + " does not exist.");
		}
	}

	@Override
	public RandomVariable getNumeraire(final MonteCarloProcess process, final double time) {
		if(discountCurveForDiscountRate != null) {
			return randomVariableFactory.createRandomVariable(1.0 / discountCurveForDiscountRate.getDiscountFactor(time));
		}
		else {
			return discountRate.mult(time).exp();
		}
	}

	@Override
	public int getNumberOfComponents() {
		return 2;
	}

	@Override
	public int getNumberOfFactors() {
		return 2;
	}

	@Override
	public RandomVariable getRandomVariableForConstant(final double value) {
		return randomVariableFactory.createRandomVariable(value);
	}

	@Override
	public SabrModel getCloneWithModifiedData(final Map<String, Object> dataModified) {
		final RandomVariableFactory newRandomVariableFactory =
				(RandomVariableFactory)dataModified.getOrDefault("randomVariableFactory", randomVariableFactory);

		final RandomVariable newInitialValue =
				RandomVariableFactory.getRandomVariableOrDefault(newRandomVariableFactory, dataModified.get("initialValue"), initialValue);
		final RandomVariable newRiskFreeRate =
				RandomVariableFactory.getRandomVariableOrDefault(newRandomVariableFactory, dataModified.get("riskFreeRate"), riskFreeRate);
		final RandomVariable newDividendYieldRate =
				RandomVariableFactory.getRandomVariableOrDefault(newRandomVariableFactory, dataModified.get("dividendYieldRate"), dividendYieldRate);
		final RandomVariable newAlpha =
				RandomVariableFactory.getRandomVariableOrDefault(newRandomVariableFactory, dataModified.get("alpha"), alpha);
		final RandomVariable newBeta =
				RandomVariableFactory.getRandomVariableOrDefault(newRandomVariableFactory, dataModified.get("beta"), beta);
		final RandomVariable newNu =
				RandomVariableFactory.getRandomVariableOrDefault(newRandomVariableFactory, dataModified.get("nu"), nu);
		final RandomVariable newRho =
				RandomVariableFactory.getRandomVariableOrDefault(newRandomVariableFactory, dataModified.get("rho"), rho);
		final RandomVariable newDiscountRate =
				RandomVariableFactory.getRandomVariableOrDefault(newRandomVariableFactory, dataModified.get("discountRate"), discountRate);

		final DiscountCurve newDiscountCurveForForwardRate =
				(DiscountCurve)dataModified.getOrDefault("discountCurveForForwardRate", discountCurveForForwardRate);
		final DiscountCurve newDiscountCurveForDividendYield =
				(DiscountCurve)dataModified.getOrDefault("discountCurveForDividendYield", discountCurveForDividendYield);
		final DiscountCurve newDiscountCurveForDiscountRate =
				(DiscountCurve)dataModified.getOrDefault("discountCurveForDiscountRate", discountCurveForDiscountRate);

		if(newDiscountCurveForForwardRate != null || newDiscountCurveForDividendYield != null || newDiscountCurveForDiscountRate != null) {
			return new SabrModel(
					newInitialValue,
					newDiscountCurveForForwardRate,
					newDiscountCurveForDividendYield != null ? newDiscountCurveForDividendYield : createFlatDiscountCurve("dividendCurve", 0.0),
					newAlpha,
					newBeta,
					newNu,
					newRho,
					newDiscountCurveForDiscountRate,
					newRandomVariableFactory
			);
		}
		else {
			return new SabrModel(
					newInitialValue,
					newRiskFreeRate,
					newDividendYieldRate != null ? newDividendYieldRate : newRandomVariableFactory.createRandomVariable(0.0),
					newAlpha,
					newBeta,
					newNu,
					newRho,
					newDiscountRate,
					newRandomVariableFactory
			);
		}
	}

	@Override
	public String toString() {
		return "SabrModel [initialValue=" + initialValue
				+ ", alpha=" + alpha
				+ ", beta=" + beta
				+ ", nu=" + nu
				+ ", rho=" + rho
				+ ", riskFreeRate=" + riskFreeRate
				+ ", dividendYieldRate=" + dividendYieldRate
				+ ", discountRate=" + discountRate
				+ "]";
	}

	public RandomVariable getInitialValue() {
		return initialValue;
	}

	public RandomVariable getAlpha() {
		return alpha;
	}

	public RandomVariable getBeta() {
		return beta;
	}

	public RandomVariable getNu() {
		return nu;
	}

	public RandomVariable getRho() {
		return rho;
	}

	public DiscountCurve getDiscountCurveForForwardRate() {
		return discountCurveForForwardRate;
	}

	public DiscountCurve getDiscountCurveForDividendYield() {
		return discountCurveForDividendYield;
	}

	public DiscountCurve getDiscountCurveForDiscountRate() {
		return discountCurveForDiscountRate;
	}

	public RandomVariable getRiskFreeRate() {
		return riskFreeRate;
	}

	public RandomVariable getDividendYieldRate() {
		return dividendYieldRate;
	}

	public RandomVariable getDiscountRate() {
		return discountRate;
	}

	private RandomVariable getRiskFreeRateAtTimeStep(final MonteCarloProcess process, final int timeIndex) {
		if(discountCurveForForwardRate != null) {
			final double time = process.getTime(timeIndex);
			final double timeNext = process.getTime(timeIndex + 1);

			final double rate = Math.log(
					discountCurveForForwardRate.getDiscountFactor(time)
					/ discountCurveForForwardRate.getDiscountFactor(timeNext)
			) / (timeNext - time);

			return randomVariableFactory.createRandomVariable(rate);
		}
		else {
			return riskFreeRate;
		}
	}

	private RandomVariable getDividendYieldRateAtTimeStep(final MonteCarloProcess process, final int timeIndex) {
		if(discountCurveForDividendYield != null) {
			final double time = process.getTime(timeIndex);
			final double timeNext = process.getTime(timeIndex + 1);

			final double rate = Math.log(
					discountCurveForDividendYield.getDiscountFactor(time)
					/ discountCurveForDividendYield.getDiscountFactor(timeNext)
			) / (timeNext - time);

			return randomVariableFactory.createRandomVariable(rate);
		}
		else {
			return dividendYieldRate;
		}
	}

	private RandomVariable powerOfSpot(final RandomVariable spot, final RandomVariable exponent) {
		return spot.log().mult(exponent).exp();
	}

	private static DiscountCurve createFlatDiscountCurve(final String name, final double rate) {
		final double[] times = new double[] { 0.0, 1.0 };
		final double[] zeroRates = new double[] { rate, rate };

		return DiscountCurveInterpolation.createDiscountCurveFromAnnualizedZeroRates(
				name,
				LocalDate.of(2010, 8, 1),
				times,
				zeroRates,
				InterpolationMethod.LINEAR,
				ExtrapolationMethod.CONSTANT,
				InterpolationEntity.VALUE
		);
	}
}