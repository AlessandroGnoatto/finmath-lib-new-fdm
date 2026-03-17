package net.finmath.montecarlo.assetderivativevaluation.myproducts;

import net.finmath.exception.CalculationException;
import net.finmath.modelling.products.BarrierType;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.montecarlo.assetderivativevaluation.AssetModelMonteCarloSimulationModel;
import net.finmath.montecarlo.assetderivativevaluation.products.AbstractAssetMonteCarloProduct;
import net.finmath.stochastic.RandomVariable;
import net.finmath.stochastic.Scalar;

/**
 * European single-barrier option.
 *
 * <p>
 * The product supports the four standard barrier types:
 * down-and-in, up-and-in, down-and-out, up-and-out.
 * </p>
 *
 * <p>
 * Rebate convention: same as in the book of Haug, page 153
 * </p>
 * <ul>
 *   <li>for knock-out options, the rebate is paid at hit,</li>
 *   <li>for knock-in options, the rebate is paid at maturity if the barrier has not been hit.</li>
 * </ul>
 *
 * <p>
 * Barrier monitoring is performed on the model time discretization.
 * </p>
 * 
 * @TODO: change package name and put it in the main library.
 *
 * @author Alessandro Gnoatto
 */
public class BarrierOption extends AbstractAssetMonteCarloProduct {

	private final double maturity;
	private final double strike;
	private final double barrier;
	private final double rebate;
	private final CallOrPut callOrPut;
	private final BarrierType barrierType;
	private final int underlyingIndex;

	public BarrierOption(
			final double maturity,
			final double strike,
			final double barrier,
			final double rebate,
			final CallOrPut callOrPut,
			final BarrierType barrierType) {
		this(maturity, strike, barrier, rebate, callOrPut, barrierType, 0);
	}

	public BarrierOption(
			final double maturity,
			final double strike,
			final double barrier,
			final double rebate,
			final CallOrPut callOrPut,
			final BarrierType barrierType,
			final int underlyingIndex) {
		super();
		this.maturity = maturity;
		this.strike = strike;
		this.barrier = barrier;
		this.rebate = rebate;
		this.callOrPut = callOrPut;
		this.barrierType = barrierType;
		this.underlyingIndex = underlyingIndex;
	}

	public BarrierOption(
			final double maturity,
			final double strike,
			final double barrier,
			final CallOrPut callOrPut,
			final BarrierType barrierType) {
		this(maturity, strike, barrier, 0.0, callOrPut, barrierType, 0);
	}

	public BarrierOption(
			final double maturity,
			final double strike,
			final double barrier,
			final CallOrPut callOrPut,
			final BarrierType barrierType,
			final int underlyingIndex) {
		this(maturity, strike, barrier, 0.0, callOrPut, barrierType, underlyingIndex);
	}

	@Override
	public RandomVariable getValue(final double evaluationTime, final AssetModelMonteCarloSimulationModel model)
			throws CalculationException {

		final int maturityIndex = model.getTimeIndex(maturity);
		if(maturityIndex < 0) {
			throw new IllegalArgumentException("Maturity " + maturity + " is not part of the model time discretization.");
		}

		final RandomVariable underlyingAtMaturity = model.getAssetValue(maturityIndex, underlyingIndex);

		final RandomVariable payoff;
		if(callOrPut == CallOrPut.CALL) {
			payoff = underlyingAtMaturity.sub(strike).floor(0.0);
		}
		else {
			payoff = underlyingAtMaturity.sub(strike).mult(-1.0).floor(0.0);
		}

		final RandomVariable value;
		switch(barrierType) {
		case DOWN_IN:
		case UP_IN:
			value = getKnockInValue(model, maturityIndex, payoff, evaluationTime);
			break;
		case DOWN_OUT:
		case UP_OUT:
			value = getKnockOutValue(model, maturityIndex, payoff, evaluationTime);
			break;
		default:
			throw new IllegalArgumentException("Unsupported barrier type: " + barrierType);
		}

		return value;
	}

	private RandomVariable getKnockInValue(
			final AssetModelMonteCarloSimulationModel model,
			final int maturityIndex,
			final RandomVariable payoff,
			final double evaluationTime) throws CalculationException {

		RandomVariable hitIndicator = new Scalar(0.0);

		for(int timeIndex = 0; timeIndex <= maturityIndex; timeIndex++) {
			final RandomVariable underlying = model.getAssetValue(timeIndex, underlyingIndex);

			final RandomVariable hitThisStep;
			switch(barrierType) {
			case DOWN_IN:
				hitThisStep = underlying.sub(barrier).choose(new Scalar(0.0), new Scalar(1.0));
				break;
			case UP_IN:
				hitThisStep = underlying.sub(barrier).choose(new Scalar(1.0), new Scalar(0.0));
				break;
			default:
				throw new IllegalArgumentException("Unsupported barrier type for knock-in payoff: " + barrierType);
			}

			hitIndicator = hitIndicator.add(hitThisStep).cap(1.0);
		}

		final RandomVariable optionPayoff =
				hitIndicator.mult(payoff).add(hitIndicator.mult(-1.0).add(1.0).mult(rebate));

		return discountToEvaluationTime(model, optionPayoff, maturity, evaluationTime);
	}

	private RandomVariable getKnockOutValue(
			final AssetModelMonteCarloSimulationModel model,
			final int maturityIndex,
			final RandomVariable payoff,
			final double evaluationTime) throws CalculationException {

		RandomVariable aliveIndicator = new Scalar(1.0);
		RandomVariable discountedRebate = new Scalar(0.0);

		for(int timeIndex = 0; timeIndex <= maturityIndex; timeIndex++) {
			final double time = model.getTime(timeIndex);
			final RandomVariable underlying = model.getAssetValue(timeIndex, underlyingIndex);

			final RandomVariable hitThisStep;
			switch(barrierType) {
			case DOWN_OUT:
				hitThisStep = underlying.sub(barrier).choose(new Scalar(0.0), new Scalar(1.0));
				break;
			case UP_OUT:
				hitThisStep = underlying.sub(barrier).choose(new Scalar(1.0), new Scalar(0.0));
				break;
			default:
				throw new IllegalArgumentException("Unsupported barrier type for knock-out payoff: " + barrierType);
			}

			final RandomVariable firstHitThisStep = aliveIndicator.mult(hitThisStep);

			if(rebate != 0.0) {
				final RandomVariable rebatePaidAtHit = firstHitThisStep.mult(rebate);
				discountedRebate = discountedRebate.add(
						discountToEvaluationTime(model, rebatePaidAtHit, time, evaluationTime));
			}

			aliveIndicator = aliveIndicator.mult(hitThisStep.mult(-1.0).add(1.0));
		}

		final RandomVariable vanillaIfAlive =
				discountToEvaluationTime(model, aliveIndicator.mult(payoff), maturity, evaluationTime);

		return vanillaIfAlive.add(discountedRebate);
	}

	private RandomVariable discountToEvaluationTime(
			final AssetModelMonteCarloSimulationModel model,
			final RandomVariable payoff,
			final double paymentTime,
			final double evaluationTime) throws CalculationException {

		final RandomVariable numeraireAtPayment = model.getNumeraire(paymentTime);
		final RandomVariable monteCarloWeightsAtPayment = model.getMonteCarloWeights(paymentTime);

		RandomVariable value = payoff.div(numeraireAtPayment).mult(monteCarloWeightsAtPayment);

		final RandomVariable numeraireAtEvaluationTime = model.getNumeraire(evaluationTime);
		final RandomVariable monteCarloWeightsAtEvaluationTime = model.getMonteCarloWeights(evaluationTime);

		value = value.mult(numeraireAtEvaluationTime).div(monteCarloWeightsAtEvaluationTime);

		return value;
	}
}