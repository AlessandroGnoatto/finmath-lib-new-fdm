package net.finmath.finitedifference.solvers;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.apache.commons.math3.analysis.interpolation.LinearInterpolator;
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction;
import org.junit.Test;

import net.finmath.exception.CalculationException;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMVarianceGammaModel;
import net.finmath.finitedifference.assetderivativevaluation.products.EuropeanOption;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.modelling.EuropeanExercise;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.montecarlo.assetderivativevaluation.AssetModelMonteCarloSimulationModel;
import net.finmath.montecarlo.assetderivativevaluation.MonteCarloVarianceGammaModel;
import net.finmath.stochastic.RandomVariable;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Monte Carlo and Fourier comparison test for {@link FDMThetaMethod1DJump}
 * under the {@link FDMVarianceGammaModel}.
 *
 * @author Alessandro Gnoatto
 */
public class FDMThetaMethod1DJumpVarianceGammaMonteCarloComparisonTest {

	private static final double INITIAL_VALUE = 100.0;
	private static final double STRIKE = 100.0;
	private static final double MATURITY = 1.0;

	private static final double RISK_FREE_RATE = 0.05;
	private static final double DIVIDEND_YIELD = 0.0;

	private static final double SIGMA = 0.30;
	private static final double THETA_PARAMETER = -0.10;
	private static final double NU = 0.20;

	private static final double LOWER_INTEGRATION_BOUND = -2.0;
	private static final double UPPER_INTEGRATION_BOUND = 2.0;

	private static final double PDE_THETA = 1.0;
	private static final int NUMBER_OF_TIME_STEPS_FD = 300;
	private static final int NUMBER_OF_SPACE_STEPS = 240;
	private static final int QUADRATURE_POINTS_PER_SIDE = 200;

	private static final int NUMBER_OF_TIME_STEPS_MC = 200;
	private static final int NUMBER_OF_PATHS = 20000;
	private static final int SEED = 3141;

	/*
	 * MC should be quite close to Fourier.
	 * FD is allowed a wider tolerance for now, since this is our first
	 * infinite-activity benchmark and truncation / quadrature effects are
	 * more delicate than in the Merton case.
	 */
	private static final double MONTE_CARLO_FOURIER_TOLERANCE = 0.35;
	private static final double FINITE_DIFFERENCE_FOURIER_TOLERANCE = 0.8;

	@Test
	public void testEuropeanCallAgainstMonteCarloAndFourierBenchmark() throws CalculationException {

		final SpaceTimeDiscretization fdDiscretization = createFiniteDifferenceDiscretization();
		final double[] sNodes = fdDiscretization.getSpaceGrid(0).getGrid();

		final EuropeanOption option = new EuropeanOption(MATURITY, STRIKE, CallOrPut.CALL);

		final FDMVarianceGammaModel fdVarianceGammaModel = new FDMVarianceGammaModel(
				INITIAL_VALUE,
				RISK_FREE_RATE,
				DIVIDEND_YIELD,
				SIGMA,
				NU,
				THETA_PARAMETER,
				LOWER_INTEGRATION_BOUND,
				UPPER_INTEGRATION_BOUND,
				fdDiscretization
		);

		final FDMThetaMethod1DJump fdSolver = new FDMThetaMethod1DJump(
				fdVarianceGammaModel,
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

		final AssetModelMonteCarloSimulationModel monteCarloVarianceGammaModel =
				createMonteCarloVarianceGammaModel();

		final double mcPrice = getMonteCarloEuropeanCallValue(
				monteCarloVarianceGammaModel,
				MATURITY,
				STRIKE
		);

		final double fourierPrice = getFourierEuropeanCallValue();

		assertTrue(fdPrice >= 0.0);
		assertTrue(mcPrice >= 0.0);
		assertTrue(fourierPrice >= 0.0);

		assertEquals(fourierPrice, mcPrice, MONTE_CARLO_FOURIER_TOLERANCE);
		assertEquals(fourierPrice, fdPrice, FINITE_DIFFERENCE_FOURIER_TOLERANCE);
	}

	private AssetModelMonteCarloSimulationModel createMonteCarloVarianceGammaModel() {
		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(
						0.0,
						NUMBER_OF_TIME_STEPS_MC,
						MATURITY / NUMBER_OF_TIME_STEPS_MC
				);

		AssetModelMonteCarloSimulationModel monteCarloVarianceGammaModel;
		{
			monteCarloVarianceGammaModel = new MonteCarloVarianceGammaModel(
					timeDiscretization,
					NUMBER_OF_PATHS,
					SEED,
					INITIAL_VALUE,
					RISK_FREE_RATE,
					SIGMA,
					THETA_PARAMETER,
					NU
			);
		}

		return monteCarloVarianceGammaModel;
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

	private double getFourierEuropeanCallValue() throws CalculationException {

		final net.finmath.fouriermethod.models.CharacteristicFunctionModel characteristicFunctionVarianceGamma =
				new net.finmath.fouriermethod.models.VarianceGammaModel(
						INITIAL_VALUE,
						RISK_FREE_RATE,
						RISK_FREE_RATE,
						SIGMA,
						THETA_PARAMETER,
						NU
				);

		final net.finmath.fouriermethod.products.FourierTransformProduct europeanFourier =
				new net.finmath.fouriermethod.products.EuropeanOption(
						MATURITY,
						STRIKE
				);

		return europeanFourier.getValue(characteristicFunctionVarianceGamma);
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
				PDE_THETA,
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