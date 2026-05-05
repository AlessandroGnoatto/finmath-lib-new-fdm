package net.finmath.finitedifference.utilities;

import java.util.function.DoubleBinaryOperator;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMBlackScholesModel;
import net.finmath.finitedifference.assetderivativevaluation.products.EuropeanOption;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Demo for finite-difference interpolation, Greek construction, and 3D plotting.
 *
 * <p>
 * The example prices a European call option under the Black-Scholes model,
 * computes value, delta, gamma, and theta surfaces, and plots each surface as a
 * function of spot and evaluation time.
 * </p>
 *
 * <p>
 * This class assumes that {@link Plot3DFXClean} is available in the test source
 * set. It deliberately keeps the production FDM utilities independent of any
 * plotting dependency.
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public final class BlackScholesSurfaceUtilitiesPlotDemo {

	private static final double MATURITY = 1.0;
	private static final double STRIKE = 100.0;
	private static final double INITIAL_SPOT = 100.0;

	private static final double RISK_FREE_RATE = 0.05;
	private static final double DIVIDEND_YIELD = 0.00;
	private static final double VOLATILITY = 0.20;

	private static final double SPOT_MIN = 1.0;
	private static final double SPOT_MAX = 220.0;

	private static final int NUMBER_OF_SPACE_STEPS = 220;
	private static final int NUMBER_OF_TIME_STEPS = 160;

	private static final double THETA = 0.5;

	private static final int NUMBER_OF_PLOT_POINTS_SPOT = 120;
	private static final int NUMBER_OF_PLOT_POINTS_TIME = 90;

	private BlackScholesSurfaceUtilitiesPlotDemo() {
	}

	/**
	 * Runs the demo.
	 *
	 * @param args Command-line arguments, ignored.
	 * @throws Exception 
	 */
	public static void main(final String[] args) throws Exception {

		final SpaceTimeDiscretization discretization = createDiscretization();

		final FDMBlackScholesModel model = new FDMBlackScholesModel(
				INITIAL_SPOT,
				RISK_FREE_RATE,
				DIVIDEND_YIELD,
				VOLATILITY,
				discretization
		);

		final EuropeanOption callOption = new EuropeanOption(
				MATURITY,
				STRIKE,
				CallOrPut.CALL
		);

		final double[][] values = callOption.getValues(model);

		runInterpolationExamples(values, discretization);

		final double[][] deltaSurface = FiniteDifferenceGreekProvider.deltaSurface(
				values,
				discretization
		);

		final double[][] gammaSurface = FiniteDifferenceGreekProvider.gammaSurface(
				values,
				discretization
		);

		final double[][] thetaSurface = FiniteDifferenceGreekProvider.thetaSurface(
				values,
				discretization
		);

		showSurfacePlot(
				values,
				discretization,
				"Black-Scholes European Call Price",
				"Spot",
				"Evaluation time",
				"Price"
		);

		showSurfacePlot(
				deltaSurface,
				discretization,
				"Black-Scholes European Call Delta",
				"Spot",
				"Evaluation time",
				"Delta"
		);

		showSurfacePlot(
				gammaSurface,
				discretization,
				"Black-Scholes European Call Gamma",
				"Spot",
				"Evaluation time",
				"Gamma"
		);

		showSurfacePlot(
				thetaSurface,
				discretization,
				"Black-Scholes European Call Theta",
				"Spot",
				"Evaluation time",
				"Theta"
		);
	}

	private static SpaceTimeDiscretization createDiscretization() {

		final Grid spotGrid = new UniformGrid(
				NUMBER_OF_SPACE_STEPS,
				SPOT_MIN,
				SPOT_MAX
		);

		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(
						0.0,
						NUMBER_OF_TIME_STEPS,
						MATURITY / NUMBER_OF_TIME_STEPS
				);

		return new SpaceTimeDiscretization(
				spotGrid,
				timeDiscretization,
				THETA,
				new double[] { INITIAL_SPOT }
		);
	}

	private static void runInterpolationExamples(
			final double[][] values,
			final SpaceTimeDiscretization discretization) {

		final FiniteDifferenceSurfaceView valueView =
				FiniteDifferenceSurfaceView.of(discretization, values);

		final int timeIndexToday =
				FiniteDifferenceValueInterpolator.getTimeIndexNearestLessOrEqual(
						discretization,
						0.0,
						MATURITY,
						values
				);

		final double valueAtInitialSpotFromTimeIndex =
				valueView.interpolate(timeIndexToday, INITIAL_SPOT);

		final double valueAtInitialSpotFromEvaluationTime =
				valueView.interpolate(
						0.0,
						MATURITY,
						INITIAL_SPOT
				);

		final double valueAtOffGridPoint =
				valueView.interpolate(
						0.37,
						MATURITY,
						103.25
				);

		final double valueNearMaturity =
				valueView.interpolate(
						0.95,
						MATURITY,
						103.25
				);

		System.out.println("Interpolation examples");
		System.out.println("----------------------");
		System.out.println("Value at t = 0.00, S = 100.00 using time index: "
				+ valueAtInitialSpotFromTimeIndex);
		System.out.println("Value at t = 0.00, S = 100.00 using evaluation time: "
				+ valueAtInitialSpotFromEvaluationTime);
		System.out.println("Value at t = 0.37, S = 103.25: "
				+ valueAtOffGridPoint);
		System.out.println("Value at t = 0.95, S = 103.25: "
				+ valueNearMaturity);
		System.out.println();
	}

	private static void showSurfacePlot(
			final double[][] surface,
			final SpaceTimeDiscretization discretization,
			final String title,
			final String xAxisLabel,
			final String yAxisLabel,
			final String zAxisLabel) throws Exception {

		final FiniteDifferencePlotData2D plotData = createSpotTimePlotData(
				surface,
				discretization,
				title,
				xAxisLabel,
				yAxisLabel,
				zAxisLabel
		);

		new Plot3DFXClean(
				plotData.getXMin(),
				plotData.getXMax(),
				plotData.getYMin(),
				plotData.getYMax(),
				plotData.getNumberOfPointsX(),
				plotData.getNumberOfPointsY(),
				plotData.getFunction()
		)
				.setTitle(plotData.getTitle())
				.setXAxisLabel(plotData.getXAxisLabel())
				.setYAxisLabel(plotData.getYAxisLabel())
				.setZAxisLabel(plotData.getZAxisLabel())
				.setIsLegendVisible(true)
				.show();
	}

	private static FiniteDifferencePlotData2D createSpotTimePlotData(
			final double[][] surface,
			final SpaceTimeDiscretization discretization,
			final String title,
			final String xAxisLabel,
			final String yAxisLabel,
			final String zAxisLabel) {

		final double[] spotGrid = discretization.getSpaceGrid(0).getGrid();

		final DoubleBinaryOperator function =
				(spot, evaluationTime) ->
						FiniteDifferenceValueInterpolator.interpolateSurface(
								surface,
								discretization,
								evaluationTime,
								MATURITY,
								spot
						);

		return new FiniteDifferencePlotData2D(
				spotGrid[0],
				spotGrid[spotGrid.length - 1],
				0.0,
				MATURITY,
				NUMBER_OF_PLOT_POINTS_SPOT,
				NUMBER_OF_PLOT_POINTS_TIME,
				function,
				title,
				xAxisLabel,
				yAxisLabel,
				zAxisLabel
		);
	}
}