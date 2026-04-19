package net.finmath.finitedifference.solvers;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.apache.commons.math3.analysis.interpolation.LinearInterpolator;
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction;
import org.junit.Test;

import net.finmath.exception.CalculationException;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMMertonModel;
import net.finmath.finitedifference.assetderivativevaluation.products.EuropeanOption;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.modelling.EuropeanExercise;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.montecarlo.assetderivativevaluation.AssetModelMonteCarloSimulationModel;
import net.finmath.montecarlo.assetderivativevaluation.MonteCarloMertonModel;
import net.finmath.stochastic.RandomVariable;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Monte Carlo comparison test for {@link FDMThetaMethod1DJump}.
 *
 * @author Alessandro Gnoatto
 */
public class FDMThetaMethod1DJumpMonteCarloComparisonTest {

	private static final double INITIAL_VALUE = 100.0;
	private static final double STRIKE = 100.0;
	private static final double MATURITY = 1.0;

	private static final double RISK_FREE_RATE = 0.05;
	private static final double DIVIDEND_YIELD = 0.0;
	private static final double VOLATILITY = 0.20;

	private static final double LAMBDA = 0.4;
	private static final double JUMP_SIZE_STD_DEV = 0.15;
	private static final double JUMP_SIZE_MEAN = Math.log(1.0);

	private static final double LOWER_INTEGRATION_BOUND = -2.0;
	private static final double UPPER_INTEGRATION_BOUND = 2.0;

	private static final double THETA = 1.0;
	private static final int NUMBER_OF_TIME_STEPS_FD = 300;
	private static final int NUMBER_OF_SPACE_STEPS = 240;
	private static final int QUADRATURE_POINTS_PER_SIDE = 200;

	private static final int NUMBER_OF_TIME_STEPS_MC = 200;
	private static final int NUMBER_OF_PATHS = 200000;
	private static final int SEED = 3141;

	private static final double MC_COMPARISON_TOLERANCE = 0.50;

	@Test
	public void testEuropeanCallAgainstMonteCarloBenchmark() throws CalculationException {

		final SpaceTimeDiscretization fdDiscretization = createFiniteDifferenceDiscretization();
		final double[] sNodes = fdDiscretization.getSpaceGrid(0).getGrid();

		final EuropeanOption option = new EuropeanOption(MATURITY, STRIKE, CallOrPut.CALL);

		final FDMMertonModel fdMertonModel = new FDMMertonModel(
				INITIAL_VALUE,
				RISK_FREE_RATE,
				DIVIDEND_YIELD,
				VOLATILITY,
				LAMBDA,
				JUMP_SIZE_MEAN,
				JUMP_SIZE_STD_DEV,
				LOWER_INTEGRATION_BOUND,
				UPPER_INTEGRATION_BOUND,
				fdDiscretization
		);

		final FDMThetaMethod1DJump fdSolver = new FDMThetaMethod1DJump(
				fdMertonModel,
				option,
				fdDiscretization,
				new EuropeanExercise(MATURITY),
				QUADRATURE_POINTS_PER_SIDE
		);

		final double[] fdValues = fdSolver.getValue(
				0.0,
				MATURITY,
				stock -> Math.max(stock - STRIKE, 0.0)
		);

		final double fdPrice = interpolateAtSpot(fdValues, sNodes, INITIAL_VALUE);

		final AssetModelMonteCarloSimulationModel monteCarloMertonModel =
				createMonteCarloMertonModel();

		final double mcPrice = getMonteCarloEuropeanCallValue(
				monteCarloMertonModel,
				MATURITY,
				STRIKE
		);

		assertTrue(fdPrice >= 0.0);
		assertTrue(mcPrice >= 0.0);

		assertEquals(mcPrice, fdPrice, MC_COMPARISON_TOLERANCE);
	}

	private AssetModelMonteCarloSimulationModel createMonteCarloMertonModel() {
		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(
						0.0,
						NUMBER_OF_TIME_STEPS_MC,
						MATURITY / NUMBER_OF_TIME_STEPS_MC
				);

		AssetModelMonteCarloSimulationModel monteCarloMertonModel;
		{
			final double m = 1.0;
			final double nu = 0.15;

			final double lambda = 0.4;
			final double jumpSizeStdDev = nu;
			final double jumpSizeMean = Math.log(m);

			monteCarloMertonModel = new MonteCarloMertonModel(
					timeDiscretization,
					NUMBER_OF_PATHS,
					SEED,
					INITIAL_VALUE,
					RISK_FREE_RATE,
					VOLATILITY,
					lambda,
					jumpSizeMean,
					jumpSizeStdDev
			);
		}

		return monteCarloMertonModel;
	}

	private double getMonteCarloEuropeanCallValue(
			final AssetModelMonteCarloSimulationModel model,
			final double maturity,
			final double strike) throws CalculationException {

		final RandomVariable underlyingAtMaturity = model.getAssetValue(maturity, 0);
		final RandomVariable payoff = underlyingAtMaturity.sub(strike).floor(0.0);

		final RandomVariable discountedPayoff =
				payoff
				.div(model.getNumeraire(maturity))
				.mult(model.getMonteCarloWeights(maturity))
				.mult(model.getNumeraire(0.0))
				.div(model.getMonteCarloWeights(0.0));

		return discountedPayoff.getAverage();
	}

	private SpaceTimeDiscretization createFiniteDifferenceDiscretization() {

		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(
						0.0,
						NUMBER_OF_TIME_STEPS_FD,
						MATURITY / NUMBER_OF_TIME_STEPS_FD
				);

		final Grid sGrid = new UniformGrid(NUMBER_OF_SPACE_STEPS, 40.0, 160.0);

		return new SpaceTimeDiscretization(
				sGrid,
				timeDiscretization,
				THETA,
				new double[] { INITIAL_VALUE }
		);
	}

	private double interpolateAtSpot(
			final double[] values,
			final double[] sNodes,
			final double spot) {

		if(spot <= sNodes[0]) {
			return values[0];
		}
		if(spot >= sNodes[sNodes.length - 1]) {
			return values[values.length - 1];
		}

		for(int i = 0; i < sNodes.length; i++) {
			if(Math.abs(sNodes[i] - spot) < 1E-12) {
				return values[i];
			}
		}

		final PolynomialSplineFunction interpolation =
				new LinearInterpolator().interpolate(sNodes, values);

		return interpolation.value(spot);
	}
}