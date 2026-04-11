package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertTrue;

import java.time.LocalDate;

import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMSabrModel;
import net.finmath.finitedifference.grids.BarrierAlignedSpotGridFactory;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.interpolation.BiLinearInterpolation;
import net.finmath.marketdata.model.curves.CurveInterpolation.ExtrapolationMethod;
import net.finmath.marketdata.model.curves.CurveInterpolation.InterpolationEntity;
import net.finmath.marketdata.model.curves.CurveInterpolation.InterpolationMethod;
import net.finmath.marketdata.model.curves.DiscountCurve;
import net.finmath.marketdata.model.curves.DiscountCurveInterpolation;
import net.finmath.modelling.products.BarrierType;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.montecarlo.BrownianMotionFromMersenneRandomNumbers;
import net.finmath.montecarlo.assetderivativevaluation.MonteCarloAssetModel;
import net.finmath.montecarlo.process.EulerSchemeFromProcessModel;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Diagnostic SABR barrier comparison test.
 *
 * <p>
 * Uses barriers closer to spot:
 * </p>
 * <ul>
 *   <li>down barrier at 90 for calls</li>
 *   <li>up barrier at 110 for puts</li>
 * </ul>
 *
 * <p>
 * Prints:
 * </p>
 * <ul>
 *   <li>FDM in / out / in+out / vanilla</li>
 *   <li>MC in / out / in+out / vanilla</li>
 *   <li>parity errors on both sides</li>
 * </ul>
 *
 * <p>
 * This is intended as a diagnostic test, not a strict pass/fail benchmark.
 * </p>
 */
public class BarrierOptionSabrMonteCarloComparisonTest {

	private static final double MATURITY = 1.0;
	private static final double STRIKE = 100.0;
	private static final double REBATE = 0.0;

	private static final double SPOT = 100.0;
	private static final double RISK_FREE_RATE = 0.02;
	private static final double DIVIDEND_YIELD = 0.01;

	private static final double INITIAL_VOLATILITY = 0.20;
	private static final double BETA = 0.50;
	private static final double NU = 0.40;
	private static final double RHO = -0.30;

	private static final double THETA = 0.5;
	private static final int FD_NUMBER_OF_TIME_STEPS = 100;
	private static final int FD_NUMBER_OF_SPACE_STEPS_S = 220;
	private static final int FD_NUMBER_OF_SPACE_STEPS_VOL = 120;
	private static final int STEPS_BETWEEN_BARRIER_AND_SPOT = 40;
	private static final double BARRIER_CLUSTERING_EXPONENT = 2.0;

	private static final int MC_NUMBER_OF_TIME_STEPS = 250;
	private static final int MC_NUMBER_OF_PATHS = 200_000;
	private static final int MC_SEED = 31415;

	@Test
	public void testDownBarrier90CallParityDecomposition() throws Exception {
		printParityDecomposition(CallOrPut.CALL, 90.0, BarrierType.DOWN_OUT, BarrierType.DOWN_IN);
	}

	@Test
	public void testUpBarrier110PutParityDecomposition() throws Exception {
		printParityDecomposition(CallOrPut.PUT, 110.0, BarrierType.UP_OUT, BarrierType.UP_IN);
	}

	private void printParityDecomposition(
			final CallOrPut callOrPut,
			final double barrier,
			final BarrierType outType,
			final BarrierType inType) throws Exception {

		final TestSetup setup = createSetup(barrier, outType);

		final BarrierOption fdmOutOption = new BarrierOption(
				MATURITY,
				STRIKE,
				barrier,
				REBATE,
				callOrPut == CallOrPut.CALL ? 1.0 : -1.0,
				outType
		);

		final BarrierOption fdmInOption = new BarrierOption(
				MATURITY,
				STRIKE,
				barrier,
				REBATE,
				callOrPut == CallOrPut.CALL ? 1.0 : -1.0,
				inType
		);

		final EuropeanOption fdmVanillaOption = new EuropeanOption(
				MATURITY,
				STRIKE,
				callOrPut == CallOrPut.CALL ? 1.0 : -1.0
		);

		final double fdmOut = interpolateAtSpotAndInitialVolatility(
				fdmOutOption.getValue(0.0, setup.fdmModel),
				setup.sNodes,
				setup.volNodes,
				SPOT,
				INITIAL_VOLATILITY
		);

		final double fdmIn = interpolateAtSpotAndInitialVolatility(
				fdmInOption.getValue(0.0, setup.fdmModel),
				setup.sNodes,
				setup.volNodes,
				SPOT,
				INITIAL_VOLATILITY
		);

		final double fdmVanilla = interpolateAtSpotAndInitialVolatility(
				fdmVanillaOption.getValue(0.0, setup.fdmModel),
				setup.sNodes,
				setup.volNodes,
				SPOT,
				INITIAL_VOLATILITY
		);

		final net.finmath.montecarlo.assetderivativevaluation.myproducts.BarrierOption mcOutOption =
				new net.finmath.montecarlo.assetderivativevaluation.myproducts.BarrierOption(
						MATURITY,
						STRIKE,
						barrier,
						REBATE,
						callOrPut,
						outType
				);

		final net.finmath.montecarlo.assetderivativevaluation.myproducts.BarrierOption mcInOption =
				new net.finmath.montecarlo.assetderivativevaluation.myproducts.BarrierOption(
						MATURITY,
						STRIKE,
						barrier,
						REBATE,
						callOrPut,
						inType
				);

		final net.finmath.montecarlo.assetderivativevaluation.products.EuropeanOption mcVanillaOption =
				new net.finmath.montecarlo.assetderivativevaluation.products.EuropeanOption(
						MATURITY,
						STRIKE
				);

		final double mcOut = mcOutOption.getValue(setup.mcModel);
		final double mcIn = mcInOption.getValue(setup.mcModel);

		final double mcVanilla;
		if(callOrPut == CallOrPut.CALL) {
			mcVanilla = mcVanillaOption.getValue(setup.mcModel);
		}
		else {
			mcVanilla = getMonteCarloEuropeanPutValue(setup.mcModel, MATURITY, STRIKE);
		}

		System.out.println("====================================================");
		System.out.println("Case                = " + callOrPut + " with barrier " + barrier);
		System.out.println("Out type            = " + outType);
		System.out.println("In type             = " + inType);
		System.out.println();

		System.out.println("FDM out             = " + fdmOut);
		System.out.println("FDM in              = " + fdmIn);
		System.out.println("FDM in + out        = " + (fdmIn + fdmOut));
		System.out.println("FDM vanilla         = " + fdmVanilla);
		System.out.println("FDM parity error    = " + Math.abs(fdmVanilla - (fdmIn + fdmOut)));
		System.out.println();

		System.out.println("MC out              = " + mcOut);
		System.out.println("MC in               = " + mcIn);
		System.out.println("MC in + out         = " + (mcIn + mcOut));
		System.out.println("MC vanilla          = " + mcVanilla);
		System.out.println("MC parity error     = " + Math.abs(mcVanilla - (mcIn + mcOut)));
		System.out.println();

		System.out.println("Abs(out FDM-MC)     = " + Math.abs(fdmOut - mcOut));
		System.out.println("Abs(in FDM-MC)      = " + Math.abs(fdmIn - mcIn));
		System.out.println("Abs(vanilla FDM-MC) = " + Math.abs(fdmVanilla - mcVanilla));
		System.out.println("====================================================");

		assertTrue("FDM out price must be non-negative.", fdmOut >= -1E-10);
		assertTrue("FDM in price must be non-negative.", fdmIn >= -1E-10);
		assertTrue("FDM vanilla price must be non-negative.", fdmVanilla >= -1E-10);

		assertTrue("MC out price must be non-negative.", mcOut >= -1E-10);
		assertTrue("MC in price must be non-negative.", mcIn >= -1E-10);
		assertTrue("MC vanilla price must be non-negative.", mcVanilla >= -1E-10);
	}

	private TestSetup createSetup(final double barrier, final BarrierType barrierType) {

		final TimeDiscretization fdTimeDiscretization = new TimeDiscretizationFromArray(
				0.0,
				FD_NUMBER_OF_TIME_STEPS,
				MATURITY / FD_NUMBER_OF_TIME_STEPS
		);

		final Grid sGrid = createSpotGrid(barrier, barrierType);

		final double volMax = Math.max(4.0 * INITIAL_VOLATILITY, 1.0);
		final Grid volGrid = new UniformGrid(
				FD_NUMBER_OF_SPACE_STEPS_VOL,
				0.0,
				volMax
		);

		final SpaceTimeDiscretization fdSpaceTime = new SpaceTimeDiscretization(
				new Grid[] { sGrid, volGrid },
				fdTimeDiscretization,
				THETA,
				new double[] { SPOT, INITIAL_VOLATILITY }
		);

		final FDMSabrModel fdmModel = new FDMSabrModel(
				SPOT,
				INITIAL_VOLATILITY,
				RISK_FREE_RATE,
				DIVIDEND_YIELD,
				BETA,
				NU,
				RHO,
				fdSpaceTime
		);

		final TimeDiscretization mcTimeDiscretization = new TimeDiscretizationFromArray(
				0.0,
				MC_NUMBER_OF_TIME_STEPS,
				MATURITY / MC_NUMBER_OF_TIME_STEPS
		);

		final BrownianMotionFromMersenneRandomNumbers brownianMotion =
				new BrownianMotionFromMersenneRandomNumbers(
						mcTimeDiscretization,
						2,
						MC_NUMBER_OF_PATHS,
						MC_SEED
				);

		final net.finmath.montecarlo.assetderivativevaluation.mymodels.SabrModel mcSabrModel =
				new net.finmath.montecarlo.assetderivativevaluation.mymodels.SabrModel(
						SPOT,
						RISK_FREE_RATE,
						DIVIDEND_YIELD,
						INITIAL_VOLATILITY,
						BETA,
						NU,
						RHO
				);

		final EulerSchemeFromProcessModel process =
				new EulerSchemeFromProcessModel(mcSabrModel, brownianMotion);

		final MonteCarloAssetModel mcModel = new MonteCarloAssetModel(process);

		return new TestSetup(
				fdmModel,
				mcModel,
				sGrid.getGrid(),
				volGrid.getGrid()
		);
	}

	private Grid createSpotGrid(final double barrier, final BarrierType barrierType) {

		final double deltaS = Math.abs(barrier - SPOT) / STEPS_BETWEEN_BARRIER_AND_SPOT;

		final boolean isKnockIn =
				barrierType == BarrierType.DOWN_IN || barrierType == BarrierType.UP_IN;

		if(barrierType == BarrierType.DOWN_IN || barrierType == BarrierType.DOWN_OUT) {

			final double sMin = Math.max(1E-8, barrier - 8.0 * deltaS);
			final double sMax = Math.max(3.0 * SPOT, SPOT + 12.0 * deltaS);

			final int numberOfSteps = Math.max(
					FD_NUMBER_OF_SPACE_STEPS_S,
					(int)Math.round((sMax - sMin) / deltaS)
			);

			if(isKnockIn) {
				return BarrierAlignedSpotGridFactory.createBarrierAlignedClusteredGrid(
						numberOfSteps,
						sMin,
						sMax,
						barrier,
						BARRIER_CLUSTERING_EXPONENT
				);
			}
			else {
				return BarrierAlignedSpotGridFactory.createBarrierAlignedUniformGrid(
						numberOfSteps,
						sMin,
						sMax,
						barrier
				);
			}
		}
		else {
			final double sMin = 0.0;
			final double sMax = barrier + 8.0 * deltaS;

			final int numberOfSteps = Math.max(
					FD_NUMBER_OF_SPACE_STEPS_S,
					(int)Math.round((sMax - sMin) / deltaS)
			);

			if(isKnockIn) {
				return BarrierAlignedSpotGridFactory.createBarrierAlignedClusteredGrid(
						numberOfSteps,
						sMin,
						sMax,
						barrier,
						BARRIER_CLUSTERING_EXPONENT
				);
			}
			else {
				return BarrierAlignedSpotGridFactory.createBarrierAlignedUniformGrid(
						numberOfSteps,
						sMin,
						sMax,
						barrier
				);
			}
		}
	}

	private double interpolateAtSpotAndInitialVolatility(
			final double[] values,
			final double[] sNodes,
			final double[] volNodes,
			final double spot,
			final double volatility) {

		assertTrue("Spot must lie inside the grid domain.",
				spot >= sNodes[0] - 1E-12 && spot <= sNodes[sNodes.length - 1] + 1E-12);
		assertTrue("Volatility must lie inside the grid domain.",
				volatility >= volNodes[0] - 1E-12 && volatility <= volNodes[volNodes.length - 1] + 1E-12);

		final int nS = sNodes.length;
		final int nVol = volNodes.length;

		final double[][] valueSurface = new double[nS][nVol];
		for(int j = 0; j < nVol; j++) {
			for(int i = 0; i < nS; i++) {
				valueSurface[i][j] = values[flatten(i, j, nS)];
			}
		}

		final BiLinearInterpolation interpolation = new BiLinearInterpolation(sNodes, volNodes, valueSurface);
		return interpolation.apply(spot, volatility);
	}

	private int flatten(final int iS, final int iVol, final int numberOfSNodes) {
		return iS + iVol * numberOfSNodes;
	}


	private double getMonteCarloEuropeanPutValue(
			final MonteCarloAssetModel model,
			final double maturity,
			final double strike) throws Exception {

		final int maturityIndex = model.getTimeIndex(maturity);
		final var underlying = model.getAssetValue(maturityIndex, 0);
		final var payoff = underlying.sub(strike).mult(-1.0).floor(0.0);

		final var numeraireAtMaturity = model.getNumeraire(maturity);
		final var weightsAtMaturity = model.getMonteCarloWeights(maturity);
		final var numeraireAtEval = model.getNumeraire(0.0);
		final var weightsAtEval = model.getMonteCarloWeights(0.0);

		return payoff.div(numeraireAtMaturity).mult(weightsAtMaturity).mult(numeraireAtEval).div(weightsAtEval).getAverage();
	}

	private static class TestSetup {

		private final FDMSabrModel fdmModel;
		private final MonteCarloAssetModel mcModel;
		private final double[] sNodes;
		private final double[] volNodes;

		private TestSetup(
				final FDMSabrModel fdmModel,
				final MonteCarloAssetModel mcModel,
				final double[] sNodes,
				final double[] volNodes) {
			this.fdmModel = fdmModel;
			this.mcModel = mcModel;
			this.sNodes = sNodes;
			this.volNodes = volNodes;
		}
	}
}