package net.finmath.finitedifference.assetderivativevaluation.products;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleBinaryOperator;

import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.solvers.FDMThetaMethod1D;
import net.finmath.finitedifference.solvers.FDMSolver;
import net.finmath.finitedifference.solvers.FDMSolverFactory;
import net.finmath.modelling.EuropeanExercise;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.modelling.products.SwingQuantityMode;
import net.finmath.modelling.products.SwingStrikeFixingConvention;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Floating-strike fixed-maturity swing option.
 *
 * <p>
 * v2 semantics:
 * </p>
 * <ul>
 *   <li>discrete decision dates,</li>
 *   <li>discrete strike-fixing dates with weights,</li>
 *   <li>local and global quantity constraints,</li>
 *   <li>bang-bang or discretized quantity-grid control,</li>
 *   <li>fix-then-exercise ordering when a fixing and a decision coincide,</li>
 *   <li>same underlying used both for payoff spot and strike fixing.</li>
 * </ul>
 *
 * <p>
 * The strike accumulator evolves on fixing dates by
 * </p>
 * <p>
 * A_new = A_old + w * S
 * </p>
 *
 * <p>
 * and the effective strike used at a decision date is
 * </p>
 * <p>
 * K(A) = strikeShift + strikeScale * A / cumulativeFixingWeight.
 * </p>
 *
 * <p>
 * The class reuses the existing 1D and 2D PDE stack by treating cumulative quantity
 * and strike accumulator as extra discrete contract states.
 * </p>
 *
 * <p>
 * As with SwingOption v1, {@link #getValues(FiniteDifferenceEquityModel)} is intentionally
 * unsupported because the contract state also depends on cumulative consumed quantity
 * and strike accumulator.
 * </p>
 * 
 * @author Alessandro Gnoatto
 */
public class FloatingStrikeSwingOption implements FiniteDifferenceProduct {

	private static final double EPS = 1.0E-12;

	private final String underlyingName;

	private final double[] decisionTimes;
	private final double[] fixingTimes;
	private final double[] fixingWeights;

	private final double maturity;

	private final double strikeShift;
	private final double strikeScale;
	private final double[] accumulatorGrid;

	private final double[] localMinQuantity;
	private final double[] localMaxQuantity;
	private final double globalMinQuantity;
	private final double globalMaxQuantity;

	private final CallOrPut callOrPut;
	private final SwingQuantityMode quantityMode;
	private final double quantityGridStep;
	private final SwingStrikeFixingConvention fixingConvention;

	public FloatingStrikeSwingOption(
			final String underlyingName,
			final double[] decisionTimes,
			final double[] fixingTimes,
			final double[] fixingWeights,
			final double strikeShift,
			final double strikeScale,
			final double[] accumulatorGrid,
			final double[] localMinQuantity,
			final double[] localMaxQuantity,
			final double globalMinQuantity,
			final double globalMaxQuantity,
			final CallOrPut callOrPut,
			final SwingQuantityMode quantityMode,
			final double quantityGridStep,
			final SwingStrikeFixingConvention fixingConvention) {

		if(decisionTimes == null || decisionTimes.length == 0) {
			throw new IllegalArgumentException("decisionTimes must contain at least one time.");
		}
		if(fixingTimes == null || fixingWeights == null || fixingTimes.length == 0) {
			throw new IllegalArgumentException("fixingTimes and fixingWeights must be non-empty.");
		}
		if(fixingTimes.length != fixingWeights.length) {
			throw new IllegalArgumentException("fixingTimes and fixingWeights must have the same length.");
		}
		if(accumulatorGrid == null || accumulatorGrid.length < 2) {
			throw new IllegalArgumentException("accumulatorGrid must contain at least two points.");
		}
		if(localMinQuantity == null || localMaxQuantity == null) {
			throw new IllegalArgumentException("Local quantity arrays must not be null.");
		}
		if(localMinQuantity.length != decisionTimes.length || localMaxQuantity.length != decisionTimes.length) {
			throw new IllegalArgumentException("Local quantity arrays must have the same length as decisionTimes.");
		}
		if(callOrPut == null) {
			throw new IllegalArgumentException("callOrPut must not be null.");
		}
		if(quantityMode == null) {
			throw new IllegalArgumentException("quantityMode must not be null.");
		}
		if(fixingConvention == null) {
			throw new IllegalArgumentException("fixingConvention must not be null.");
		}
		if(quantityMode == SwingQuantityMode.DISCRETE_QUANTITY_GRID && quantityGridStep <= 0.0) {
			throw new IllegalArgumentException("quantityGridStep must be positive for DISCRETE_QUANTITY_GRID mode.");
		}
		if(fixingConvention != SwingStrikeFixingConvention.FIX_THEN_EXERCISE) {
			throw new IllegalArgumentException("Only FIX_THEN_EXERCISE is currently supported.");
		}

		this.underlyingName = underlyingName;
		this.decisionTimes = decisionTimes.clone();
		this.fixingTimes = fixingTimes.clone();
		this.fixingWeights = fixingWeights.clone();
		this.maturity = Math.max(
				decisionTimes[decisionTimes.length - 1],
				fixingTimes[fixingTimes.length - 1]
		);
		this.strikeShift = strikeShift;
		this.strikeScale = strikeScale;
		this.accumulatorGrid = accumulatorGrid.clone();
		this.localMinQuantity = localMinQuantity.clone();
		this.localMaxQuantity = localMaxQuantity.clone();
		this.globalMinQuantity = globalMinQuantity;
		this.globalMaxQuantity = globalMaxQuantity;
		this.callOrPut = callOrPut;
		this.quantityMode = quantityMode;
		this.quantityGridStep = quantityGridStep;
		this.fixingConvention = fixingConvention;

		validateInputs();
	}

	public FloatingStrikeSwingOption(
			final double[] decisionTimes,
			final double[] fixingTimes,
			final double[] fixingWeights,
			final double strikeShift,
			final double strikeScale,
			final double[] accumulatorGrid,
			final double[] localMinQuantity,
			final double[] localMaxQuantity,
			final double globalMinQuantity,
			final double globalMaxQuantity,
			final CallOrPut callOrPut,
			final SwingQuantityMode quantityMode,
			final double quantityGridStep,
			final SwingStrikeFixingConvention fixingConvention) {
		this(
				null,
				decisionTimes,
				fixingTimes,
				fixingWeights,
				strikeShift,
				strikeScale,
				accumulatorGrid,
				localMinQuantity,
				localMaxQuantity,
				globalMinQuantity,
				globalMaxQuantity,
				callOrPut,
				quantityMode,
				quantityGridStep,
				fixingConvention
		);
	}

	public FloatingStrikeSwingOption(
			final String underlyingName,
			final double[] decisionTimes,
			final double[] fixingTimes,
			final double[] fixingWeights,
			final double strikeShift,
			final double strikeScale,
			final double[] accumulatorGrid,
			final double localMinQuantity,
			final double localMaxQuantity,
			final double globalMinQuantity,
			final double globalMaxQuantity,
			final CallOrPut callOrPut,
			final SwingQuantityMode quantityMode,
			final double quantityGridStep,
			final SwingStrikeFixingConvention fixingConvention) {
		this(
				underlyingName,
				decisionTimes,
				fixingTimes,
				fixingWeights,
				strikeShift,
				strikeScale,
				accumulatorGrid,
				fill(decisionTimes.length, localMinQuantity),
				fill(decisionTimes.length, localMaxQuantity),
				globalMinQuantity,
				globalMaxQuantity,
				callOrPut,
				quantityMode,
				quantityGridStep,
				fixingConvention
		);
	}

	public FloatingStrikeSwingOption(
			final double[] decisionTimes,
			final double[] fixingTimes,
			final double[] fixingWeights,
			final double strikeShift,
			final double strikeScale,
			final double[] accumulatorGrid,
			final double localMinQuantity,
			final double localMaxQuantity,
			final double globalMinQuantity,
			final double globalMaxQuantity,
			final CallOrPut callOrPut,
			final SwingQuantityMode quantityMode,
			final double quantityGridStep,
			final SwingStrikeFixingConvention fixingConvention) {
		this(
				null,
				decisionTimes,
				fixingTimes,
				fixingWeights,
				strikeShift,
				strikeScale,
				accumulatorGrid,
				localMinQuantity,
				localMaxQuantity,
				globalMinQuantity,
				globalMaxQuantity,
				callOrPut,
				quantityMode,
				quantityGridStep,
				fixingConvention
		);
	}

	@Override
	public double[] getValue(final double evaluationTime, final FiniteDifferenceEquityModel model) {

		if(Math.abs(evaluationTime) > EPS) {
			throw new UnsupportedOperationException(
					"FloatingStrikeSwingOption v2 supports getValue only at evaluationTime = 0."
			);
		}
		if(model == null) {
			throw new IllegalArgumentException("model must not be null.");
		}

		final SpaceTimeDiscretization baseDiscretization = model.getSpaceTimeDiscretization();
		final int dimensions = baseDiscretization.getNumberOfSpaceGrids();

		if(dimensions != 1 && dimensions != 2) {
			throw new IllegalArgumentException("FloatingStrikeSwingOption currently supports only 1D and 2D models.");
		}

		if(globalMaxQuantity <= EPS || getTotalLocalMaxQuantity() <= EPS) {
			return new double[getNumberOfStates(baseDiscretization)];
		}

		final List<double[]> cumulativeQuantityGrids = buildCumulativeQuantityGrids();
		final List<Event> events = buildEvents();

		List<List<double[]>> nextEventValuePlanes = buildZeroValuePlanes(
				cumulativeQuantityGrids.get(cumulativeQuantityGrids.size() - 1).length,
				accumulatorGrid.length,
				getNumberOfStates(baseDiscretization)
		);

		double nextEventTime = maturity;

		for(int eventIndex = events.size() - 1; eventIndex >= 0; eventIndex--) {
			final Event event = events.get(eventIndex);

			final double segmentLength = nextEventTime - event.time;
			List<List<double[]>> valueAfterCurrentEvent = propagateBackwardOverSegment(
					model,
					segmentLength,
					nextEventValuePlanes
			);

			if(event.hasDecision()) {
				valueAfterCurrentEvent = applyDecisionAtEvent(
						event,
						baseDiscretization,
						cumulativeQuantityGrids,
						valueAfterCurrentEvent
				);
			}

			if(event.hasFixing()) {
				valueAfterCurrentEvent = applyFixingAtEvent(
						baseDiscretization,
						event.fixingWeightAtDate,
						valueAfterCurrentEvent
				);
			}

			nextEventValuePlanes = valueAfterCurrentEvent;
			nextEventTime = event.time;
		}

		if(nextEventTime > EPS) {
			nextEventValuePlanes = propagateBackwardOverSegment(
					model,
					nextEventTime,
					nextEventValuePlanes
			);
		}

		if(nextEventValuePlanes.size() != 1) {
			throw new IllegalStateException("Internal error: initial quantity plane should be unique.");
		}

		return interpolateAccumulatorPlaneSlice(nextEventValuePlanes.get(0), 0.0);
	}

	@Override
	public double[][] getValues(final FiniteDifferenceEquityModel model) {
		throw new UnsupportedOperationException(
				"FloatingStrikeSwingOption carries additional discrete states "
				+ "(cumulative quantity and strike accumulator). "
				+ "Use getValue(0.0, model) in this v2 implementation."
		);
	}

	private List<List<double[]>> applyDecisionAtEvent(
			final Event event,
			final SpaceTimeDiscretization discretization,
			final List<double[]> cumulativeQuantityGrids,
			final List<List<double[]>> valueAfterDecision) {

		final double[] previousQuantityGrid = cumulativeQuantityGrids.get(event.decisionIndex);
		final double[] nextQuantityGrid = cumulativeQuantityGrids.get(event.decisionIndex + 1);
		final int numberOfStates = getNumberOfStates(discretization);

		final List<List<double[]>> valueBeforeDecision = new ArrayList<>(previousQuantityGrid.length);

		for(int previousIndex = 0; previousIndex < previousQuantityGrid.length; previousIndex++) {

			final double previousQuantity = previousQuantityGrid[previousIndex];
			final int[] admissibleDestinationIndices = getAdmissibleDestinationIndices(
					event.decisionIndex,
					previousQuantity,
					nextQuantityGrid
			);

			if(admissibleDestinationIndices.length == 0) {
				throw new IllegalStateException(
						"No admissible next quantities found from previous cumulative quantity " + previousQuantity
				);
			}

			final List<double[]> accumulatorSlices = new ArrayList<>(accumulatorGrid.length);

			for(int accumulatorIndex = 0; accumulatorIndex < accumulatorGrid.length; accumulatorIndex++) {

				final double effectiveStrike = getEffectiveStrike(
						accumulatorGrid[accumulatorIndex],
						event.cumulativeWeightAfterFixing
				);

				final double[] unitIntrinsic = buildUnitIntrinsicVector(discretization, effectiveStrike);
				final double[] valueSlice = new double[numberOfStates];
				Arrays.fill(valueSlice, Double.NEGATIVE_INFINITY);

				for(final int nextIndex : admissibleDestinationIndices) {
					final double exercisedQuantity = nextQuantityGrid[nextIndex] - previousQuantity;
					final double[] continuation = valueAfterDecision.get(nextIndex).get(accumulatorIndex);

					for(int stateIndex = 0; stateIndex < numberOfStates; stateIndex++) {
						final double candidate =
								exercisedQuantity * unitIntrinsic[stateIndex] + continuation[stateIndex];

						if(candidate > valueSlice[stateIndex]) {
							valueSlice[stateIndex] = candidate;
						}
					}
				}

				accumulatorSlices.add(valueSlice);
			}

			valueBeforeDecision.add(accumulatorSlices);
		}

		return valueBeforeDecision;
	}

	private List<List<double[]>> applyFixingAtEvent(
			final SpaceTimeDiscretization discretization,
			final double fixingWeightAtDate,
			final List<List<double[]>> valueAfterFixing) {

		final int dimensions = discretization.getNumberOfSpaceGrids();
		final double[] x0Grid = discretization.getSpaceGrid(0).getGrid();

		final List<List<double[]>> valueBeforeFixing = new ArrayList<>(valueAfterFixing.size());

		if(dimensions == 1) {
			for(int quantityIndex = 0; quantityIndex < valueAfterFixing.size(); quantityIndex++) {
				final List<double[]> oldAccumulatorSlices = new ArrayList<>(accumulatorGrid.length);

				for(int oldAccumulatorIndex = 0; oldAccumulatorIndex < accumulatorGrid.length; oldAccumulatorIndex++) {
					final double oldAccumulator = accumulatorGrid[oldAccumulatorIndex];
					final double[] out = new double[x0Grid.length];

					for(int i = 0; i < x0Grid.length; i++) {
						final double queryAccumulator = oldAccumulator + fixingWeightAtDate * x0Grid[i];
						out[i] = interpolateAccumulatorAtState(
								valueAfterFixing.get(quantityIndex),
								queryAccumulator,
								i
						);
					}

					oldAccumulatorSlices.add(out);
				}

				valueBeforeFixing.add(oldAccumulatorSlices);
			}
		}
		else if(dimensions == 2) {
			final double[] x1Grid = discretization.getSpaceGrid(1).getGrid();
			final int numberOfStates = x0Grid.length * x1Grid.length;

			for(int quantityIndex = 0; quantityIndex < valueAfterFixing.size(); quantityIndex++) {
				final List<double[]> oldAccumulatorSlices = new ArrayList<>(accumulatorGrid.length);

				for(int oldAccumulatorIndex = 0; oldAccumulatorIndex < accumulatorGrid.length; oldAccumulatorIndex++) {
					final double oldAccumulator = accumulatorGrid[oldAccumulatorIndex];
					final double[] out = new double[numberOfStates];

					for(int j = 0; j < x1Grid.length; j++) {
						for(int i = 0; i < x0Grid.length; i++) {
							final int flat = flatten(i, j, x0Grid.length);
							final double queryAccumulator = oldAccumulator + fixingWeightAtDate * x0Grid[i];

							out[flat] = interpolateAccumulatorAtState(
									valueAfterFixing.get(quantityIndex),
									queryAccumulator,
									flat
							);
						}
					}

					oldAccumulatorSlices.add(out);
				}

				valueBeforeFixing.add(oldAccumulatorSlices);
			}
		}
		else {
			throw new IllegalArgumentException("Only 1D and 2D models are supported.");
		}

		return valueBeforeFixing;
	}

	private List<List<double[]>> propagateBackwardOverSegment(
			final FiniteDifferenceEquityModel model,
			final double segmentLength,
			final List<List<double[]>> terminalPlanesAtSegmentEnd) {

		if(segmentLength <= EPS) {
			return deepCopyValuePlanes(terminalPlanesAtSegmentEnd);
		}

		final SpaceTimeDiscretization segmentDiscretization = createSegmentDiscretization(
				model.getSpaceTimeDiscretization(),
				segmentLength
		);

		final FiniteDifferenceEquityModel segmentModel =
				model.getCloneWithModifiedSpaceTimeDiscretization(segmentDiscretization);

		final FiniteDifferenceProduct proxyProduct =
				new EuropeanOption(underlyingName, segmentLength, strikeShift, callOrPut);

		final int dimensions = segmentDiscretization.getNumberOfSpaceGrids();
		final List<List<double[]>> propagated = new ArrayList<>(terminalPlanesAtSegmentEnd.size());

		if(dimensions == 1) {
			final FDMThetaMethod1D solver = new FDMThetaMethod1D(
					segmentModel,
					proxyProduct,
					segmentDiscretization,
					new EuropeanExercise(segmentLength)
			);

			for(final List<double[]> accumulatorPlanes : terminalPlanesAtSegmentEnd) {
				final List<double[]> propagatedAccumulatorPlanes = new ArrayList<>(accumulatorPlanes.size());

				for(final double[] terminalSlice : accumulatorPlanes) {
					propagatedAccumulatorPlanes.add(solver.getValue(0.0, segmentLength, terminalSlice));
				}

				propagated.add(propagatedAccumulatorPlanes);
			}
		}
		else if(dimensions == 2) {
			final FDMSolver solver = FDMSolverFactory.createSolver(
					segmentModel,
					proxyProduct,
					segmentDiscretization,
					new EuropeanExercise(segmentLength)
			);

			final double[] x0Grid = segmentDiscretization.getSpaceGrid(0).getGrid();
			final double[] x1Grid = segmentDiscretization.getSpaceGrid(1).getGrid();

			for(final List<double[]> accumulatorPlanes : terminalPlanesAtSegmentEnd) {
				final List<double[]> propagatedAccumulatorPlanes = new ArrayList<>(accumulatorPlanes.size());

				for(final double[] terminalSlice : accumulatorPlanes) {
					final DoubleBinaryOperator terminalFunction =
							(x0, x1) -> interpolate2DWithConstantExtrapolation(
									terminalSlice,
									x0Grid,
									x1Grid,
									x0,
									x1
							);

					propagatedAccumulatorPlanes.add(solver.getValue(0.0, segmentLength, terminalFunction));
				}

				propagated.add(propagatedAccumulatorPlanes);
			}
		}
		else {
			throw new IllegalArgumentException("Only 1D and 2D models are supported.");
		}

		return propagated;
	}

	private List<double[]> buildCumulativeQuantityGrids() {

		final int numberOfDecisionTimes = decisionTimes.length;

		final double[] suffixMin = new double[numberOfDecisionTimes + 1];
		final double[] suffixMax = new double[numberOfDecisionTimes + 1];

		for(int i = numberOfDecisionTimes - 1; i >= 0; i--) {
			suffixMin[i] = suffixMin[i + 1] + localMinQuantity[i];
			suffixMax[i] = suffixMax[i + 1] + localMaxQuantity[i];
		}

		final List<double[]> grids = new ArrayList<>(numberOfDecisionTimes + 1);
		grids.add(new double[] { 0.0 });

		for(int decisionIndex = 0; decisionIndex < numberOfDecisionTimes; decisionIndex++) {
			final double[] previousGrid = grids.get(decisionIndex);
			final double[] localQuantityCandidates = buildLocalQuantityCandidates(decisionIndex);

			final List<Double> nextGridCandidates = new ArrayList<>();

			for(final double previousQuantity : previousGrid) {
				for(final double exercisedQuantity : localQuantityCandidates) {

					final double nextQuantity = previousQuantity + exercisedQuantity;

					final boolean canStillReachGlobalMinimum =
							nextQuantity + suffixMax[decisionIndex + 1] >= globalMinQuantity - EPS;

					final boolean canStillRespectGlobalMaximum =
							nextQuantity + suffixMin[decisionIndex + 1] <= globalMaxQuantity + EPS;

					if(canStillReachGlobalMinimum && canStillRespectGlobalMaximum) {
						nextGridCandidates.add(nextQuantity);
					}
				}
			}

			final double[] nextGrid = uniqueSorted(nextGridCandidates);
			if(nextGrid.length == 0) {
				throw new IllegalArgumentException(
						"No feasible cumulative quantity states found at decision index " + decisionIndex
				);
			}

			grids.add(nextGrid);
		}

		return grids;
	}

	private double[] buildLocalQuantityCandidates(final int decisionIndex) {

		final double localMin = localMinQuantity[decisionIndex];
		final double localMax = localMaxQuantity[decisionIndex];

		if(quantityMode == SwingQuantityMode.BANG_BANG) {
			if(Math.abs(localMax - localMin) < EPS) {
				return new double[] { localMin };
			}
			return new double[] { localMin, localMax };
		}

		final List<Double> candidates = new ArrayList<>();
		double current = localMin;

		while(current < localMax - EPS) {
			candidates.add(current);
			current += quantityGridStep;
		}
		candidates.add(localMax);

		return uniqueSorted(candidates);
	}

	private List<Event> buildEvents() {

		final List<Event> events = new ArrayList<>();

		int fixingPointer = 0;
		int decisionPointer = 0;
		double cumulativeWeight = 0.0;

		while(fixingPointer < fixingTimes.length || decisionPointer < decisionTimes.length) {

			final double nextFixingTime =
					fixingPointer < fixingTimes.length ? fixingTimes[fixingPointer] : Double.POSITIVE_INFINITY;

			final double nextDecisionTime =
					decisionPointer < decisionTimes.length ? decisionTimes[decisionPointer] : Double.POSITIVE_INFINITY;

			final double eventTime = Math.min(nextFixingTime, nextDecisionTime);

			double fixingWeightAtDate = 0.0;
			while(fixingPointer < fixingTimes.length && Math.abs(fixingTimes[fixingPointer] - eventTime) < EPS) {
				fixingWeightAtDate += fixingWeights[fixingPointer];
				fixingPointer++;
			}

			final boolean hasDecision =
					decisionPointer < decisionTimes.length
					&& Math.abs(decisionTimes[decisionPointer] - eventTime) < EPS;

			final int decisionIndex = hasDecision ? decisionPointer : -1;
			if(hasDecision) {
				decisionPointer++;
			}

			cumulativeWeight += fixingWeightAtDate;

			events.add(new Event(
					eventTime,
					fixingWeightAtDate,
					decisionIndex,
					cumulativeWeight
			));
		}

		return events;
	}

	private List<List<double[]>> buildZeroValuePlanes(
			final int numberOfQuantityStates,
			final int numberOfAccumulatorStates,
			final int numberOfModelStates) {

		final List<List<double[]>> surfaces = new ArrayList<>(numberOfQuantityStates);

		for(int quantityIndex = 0; quantityIndex < numberOfQuantityStates; quantityIndex++) {
			final List<double[]> accumulatorPlanes = new ArrayList<>(numberOfAccumulatorStates);

			for(int accumulatorIndex = 0; accumulatorIndex < numberOfAccumulatorStates; accumulatorIndex++) {
				accumulatorPlanes.add(new double[numberOfModelStates]);
			}

			surfaces.add(accumulatorPlanes);
		}

		return surfaces;
	}

	private double[] interpolateAccumulatorPlaneSlice(
			final List<double[]> accumulatorPlanes,
			final double accumulatorQuery) {

		final int numberOfStates = accumulatorPlanes.get(0).length;
		final double[] out = new double[numberOfStates];

		for(int stateIndex = 0; stateIndex < numberOfStates; stateIndex++) {
			out[stateIndex] = interpolateAccumulatorAtState(
					accumulatorPlanes,
					accumulatorQuery,
					stateIndex
			);
		}

		return out;
	}

	private double interpolateAccumulatorAtState(
			final List<double[]> accumulatorPlanes,
			final double accumulatorQuery,
			final int stateIndex) {

		final int lowerIndex = getLowerBracketIndexWithConstantExtrapolation(accumulatorGrid, accumulatorQuery);
		final int upperIndex = Math.min(lowerIndex + 1, accumulatorGrid.length - 1);

		final double aL = accumulatorGrid[lowerIndex];
		final double aU = accumulatorGrid[upperIndex];

		final double vL = accumulatorPlanes.get(lowerIndex)[stateIndex];
		final double vU = accumulatorPlanes.get(upperIndex)[stateIndex];

		if(lowerIndex == upperIndex || Math.abs(aU - aL) < EPS) {
			return vL;
		}

		final double w = (accumulatorQuery - aL) / (aU - aL);
		return (1.0 - w) * vL + w * vU;
	}

	private int[] getAdmissibleDestinationIndices(
			final int decisionIndex,
			final double previousQuantity,
			final double[] nextQuantityGrid) {

		final double localMin = localMinQuantity[decisionIndex];
		final double localMax = localMaxQuantity[decisionIndex];

		final List<Integer> indices = new ArrayList<>();

		for(int nextIndex = 0; nextIndex < nextQuantityGrid.length; nextIndex++) {
			final double exercisedQuantity = nextQuantityGrid[nextIndex] - previousQuantity;

			if(exercisedQuantity >= localMin - EPS
					&& exercisedQuantity <= localMax + EPS) {
				indices.add(nextIndex);
			}
		}

		final int[] result = new int[indices.size()];
		for(int i = 0; i < indices.size(); i++) {
			result[i] = indices.get(i);
		}
		return result;
	}

	private double getEffectiveStrike(final double accumulatorValue, final double cumulativeWeightAtDecision) {
		if(Math.abs(strikeScale) < EPS) {
			return strikeShift;
		}
		if(cumulativeWeightAtDecision <= EPS) {
			throw new IllegalArgumentException(
					"Positive cumulative fixing weight is required at each decision date for floating strike."
			);
		}
		return strikeShift + strikeScale * accumulatorValue / cumulativeWeightAtDecision;
	}

	private double[] buildUnitIntrinsicVector(
			final SpaceTimeDiscretization discretization,
			final double effectiveStrike) {

		final double[] x0Grid = discretization.getSpaceGrid(0).getGrid();
		final int dimensions = discretization.getNumberOfSpaceGrids();

		if(dimensions == 1) {
			final double[] payoff = new double[x0Grid.length];
			for(int i = 0; i < x0Grid.length; i++) {
				payoff[i] = unitIntrinsic(x0Grid[i], effectiveStrike);
			}
			return payoff;
		}
		else if(dimensions == 2) {
			final double[] x1Grid = discretization.getSpaceGrid(1).getGrid();
			final double[] payoff = new double[x0Grid.length * x1Grid.length];

			for(int j = 0; j < x1Grid.length; j++) {
				for(int i = 0; i < x0Grid.length; i++) {
					payoff[flatten(i, j, x0Grid.length)] = unitIntrinsic(x0Grid[i], effectiveStrike);
				}
			}
			return payoff;
		}
		else {
			throw new IllegalArgumentException("Only 1D and 2D models are supported.");
		}
	}

	private double unitIntrinsic(final double assetValue, final double effectiveStrike) {
		final double signedIntrinsic = callOrPut.toInteger() * (assetValue - effectiveStrike);
		return Math.max(signedIntrinsic, 0.0);
	}

	private SpaceTimeDiscretization createSegmentDiscretization(
			final SpaceTimeDiscretization baseDiscretization,
			final double segmentLength) {

		final TimeDiscretization baseTimeDiscretization = baseDiscretization.getTimeDiscretization();
		final int baseNumberOfTimeSteps = baseTimeDiscretization.getNumberOfTimeSteps();
		final double baseLastTime = Math.max(baseTimeDiscretization.getLastTime(), EPS);

		final int segmentNumberOfTimeSteps = Math.max(
				1,
				(int)Math.round(baseNumberOfTimeSteps * segmentLength / baseLastTime)
		);

		final TimeDiscretization segmentTimeDiscretization = new TimeDiscretizationFromArray(
				0.0,
				segmentNumberOfTimeSteps,
				segmentLength / segmentNumberOfTimeSteps
		);

		if(baseDiscretization.getNumberOfSpaceGrids() == 1) {
			return new SpaceTimeDiscretization(
					baseDiscretization.getSpaceGrid(0),
					segmentTimeDiscretization,
					baseDiscretization.getTheta(),
					new double[] { baseDiscretization.getCenter(0) }
			);
		}

		final int numberOfSpaceGrids = baseDiscretization.getNumberOfSpaceGrids();
		final Grid[] spaceGrids = new Grid[numberOfSpaceGrids];
		final double[] center = new double[numberOfSpaceGrids];

		for(int i = 0; i < numberOfSpaceGrids; i++) {
			spaceGrids[i] = baseDiscretization.getSpaceGrid(i);
			center[i] = baseDiscretization.getCenter(i);
		}

		return new SpaceTimeDiscretization(
				spaceGrids,
				segmentTimeDiscretization,
				baseDiscretization.getTheta(),
				center
		);
	}

	private int getNumberOfStates(final SpaceTimeDiscretization discretization) {
		if(discretization.getNumberOfSpaceGrids() == 1) {
			return discretization.getSpaceGrid(0).getGrid().length;
		}
		return discretization.getSpaceGrid(0).getGrid().length
				* discretization.getSpaceGrid(1).getGrid().length;
	}

	private double interpolate2DWithConstantExtrapolation(
			final double[] flattenedValues,
			final double[] x0Grid,
			final double[] x1Grid,
			final double x0Query,
			final double x1Query) {

		final int i0 = getLowerBracketIndexWithConstantExtrapolation(x0Grid, x0Query);
		final int i1 = Math.min(i0 + 1, x0Grid.length - 1);

		final int j0 = getLowerBracketIndexWithConstantExtrapolation(x1Grid, x1Query);
		final int j1 = Math.min(j0 + 1, x1Grid.length - 1);

		final double x0L = x0Grid[i0];
		final double x0U = x0Grid[i1];
		final double x1L = x1Grid[j0];
		final double x1U = x1Grid[j1];

		final double f00 = flattenedValues[flatten(i0, j0, x0Grid.length)];
		final double f10 = flattenedValues[flatten(i1, j0, x0Grid.length)];
		final double f01 = flattenedValues[flatten(i0, j1, x0Grid.length)];
		final double f11 = flattenedValues[flatten(i1, j1, x0Grid.length)];

		final double wx = (i0 == i1 || Math.abs(x0U - x0L) < EPS) ? 0.0 : (x0Query - x0L) / (x0U - x0L);
		final double wy = (j0 == j1 || Math.abs(x1U - x1L) < EPS) ? 0.0 : (x1Query - x1L) / (x1U - x1L);

		return (1.0 - wx) * (1.0 - wy) * f00
				+ wx * (1.0 - wy) * f10
				+ (1.0 - wx) * wy * f01
				+ wx * wy * f11;
	}

	private int getLowerBracketIndexWithConstantExtrapolation(final double[] grid, final double x) {
		if(x <= grid[0]) {
			return 0;
		}
		if(x >= grid[grid.length - 1]) {
			return grid.length - 2;
		}

		int upperIndex = 1;
		while(upperIndex < grid.length && grid[upperIndex] < x) {
			upperIndex++;
		}
		return upperIndex - 1;
	}

	private int flatten(final int i0, final int i1, final int n0) {
		return i0 + i1 * n0;
	}

	private List<List<double[]>> deepCopyValuePlanes(final List<List<double[]>> planes) {
		final List<List<double[]>> copy = new ArrayList<>(planes.size());

		for(final List<double[]> accumulatorPlanes : planes) {
			final List<double[]> accumulatorCopy = new ArrayList<>(accumulatorPlanes.size());

			for(final double[] slice : accumulatorPlanes) {
				accumulatorCopy.add(slice.clone());
			}

			copy.add(accumulatorCopy);
		}

		return copy;
	}

	private double[] uniqueSorted(final List<Double> values) {
		if(values.isEmpty()) {
			return new double[0];
		}

		final double[] raw = new double[values.size()];
		for(int i = 0; i < values.size(); i++) {
			raw[i] = values.get(i);
		}
		Arrays.sort(raw);

		final List<Double> unique = new ArrayList<>();
		unique.add(raw[0]);

		for(int i = 1; i < raw.length; i++) {
			if(Math.abs(raw[i] - unique.get(unique.size() - 1)) > 1.0E-10) {
				unique.add(raw[i]);
			}
		}

		final double[] result = new double[unique.size()];
		for(int i = 0; i < unique.size(); i++) {
			result[i] = unique.get(i);
		}
		return result;
	}

	private double getTotalLocalMaxQuantity() {
		double total = 0.0;
		for(final double q : localMaxQuantity) {
			total += q;
		}
		return total;
	}

	private void validateInputs() {

		for(int i = 0; i < decisionTimes.length; i++) {
			if(decisionTimes[i] < 0.0) {
				throw new IllegalArgumentException("decisionTimes must be non-negative.");
			}
			if(i > 0 && decisionTimes[i] <= decisionTimes[i - 1]) {
				throw new IllegalArgumentException("decisionTimes must be strictly increasing.");
			}
			if(localMinQuantity[i] < 0.0 || localMaxQuantity[i] < 0.0) {
				throw new IllegalArgumentException("Local quantities must be non-negative.");
			}
			if(localMaxQuantity[i] + EPS < localMinQuantity[i]) {
				throw new IllegalArgumentException("Each localMaxQuantity must be >= localMinQuantity.");
			}
		}

		for(int i = 0; i < fixingTimes.length; i++) {
			if(fixingTimes[i] < 0.0) {
				throw new IllegalArgumentException("fixingTimes must be non-negative.");
			}
			if(i > 0 && fixingTimes[i] < fixingTimes[i - 1] - EPS) {
				throw new IllegalArgumentException("fixingTimes must be non-decreasing.");
			}
			if(fixingWeights[i] < 0.0) {
				throw new IllegalArgumentException("fixingWeights must be non-negative.");
			}
		}

		for(int i = 0; i < accumulatorGrid.length; i++) {
			if(i > 0 && accumulatorGrid[i] <= accumulatorGrid[i - 1]) {
				throw new IllegalArgumentException("accumulatorGrid must be strictly increasing.");
			}
		}
		if(accumulatorGrid[0] > EPS) {
			throw new IllegalArgumentException("accumulatorGrid should contain or lie below the initial accumulator 0.");
		}

		if(globalMinQuantity < 0.0 || globalMaxQuantity < 0.0) {
			throw new IllegalArgumentException("Global quantities must be non-negative.");
		}
		if(globalMaxQuantity + EPS < globalMinQuantity) {
			throw new IllegalArgumentException("globalMaxQuantity must be >= globalMinQuantity.");
		}

		double totalLocalMin = 0.0;
		double totalLocalMax = 0.0;
		for(int i = 0; i < localMinQuantity.length; i++) {
			totalLocalMin += localMinQuantity[i];
			totalLocalMax += localMaxQuantity[i];
		}

		if(totalLocalMin > globalMaxQuantity + EPS) {
			throw new IllegalArgumentException("Global maximum is below the mandatory total local minimum.");
		}
		if(totalLocalMax + EPS < globalMinQuantity) {
			throw new IllegalArgumentException("Global minimum is above the achievable total local maximum.");
		}

		double cumulativeFixingWeight = 0.0;
		int fixingPointer = 0;

		for(final double decisionTime : decisionTimes) {
			while(fixingPointer < fixingTimes.length && fixingTimes[fixingPointer] <= decisionTime + EPS) {
				cumulativeFixingWeight += fixingWeights[fixingPointer];
				fixingPointer++;
			}

			if(Math.abs(strikeScale) > EPS && cumulativeFixingWeight <= EPS) {
				throw new IllegalArgumentException(
						"Each decision time must have positive cumulative fixing weight when strikeScale != 0."
				);
			}
		}
	}

	public String getUnderlyingName() {
		return underlyingName;
	}

	public double[] getDecisionTimes() {
		return decisionTimes.clone();
	}

	public double[] getFixingTimes() {
		return fixingTimes.clone();
	}

	public double[] getFixingWeights() {
		return fixingWeights.clone();
	}

	public double getMaturity() {
		return maturity;
	}

	public double getStrikeShift() {
		return strikeShift;
	}

	public double getStrikeScale() {
		return strikeScale;
	}

	public double[] getAccumulatorGrid() {
		return accumulatorGrid.clone();
	}

	public double[] getLocalMinQuantity() {
		return localMinQuantity.clone();
	}

	public double[] getLocalMaxQuantity() {
		return localMaxQuantity.clone();
	}

	public double getGlobalMinQuantity() {
		return globalMinQuantity;
	}

	public double getGlobalMaxQuantity() {
		return globalMaxQuantity;
	}

	public CallOrPut getCallOrPut() {
		return callOrPut;
	}

	public SwingQuantityMode getQuantityMode() {
		return quantityMode;
	}

	public double getQuantityGridStep() {
		return quantityGridStep;
	}

	public SwingStrikeFixingConvention getFixingConvention() {
		return fixingConvention;
	}

	private static final class Event {

		private final double time;
		private final double fixingWeightAtDate;
		private final int decisionIndex;
		private final double cumulativeWeightAfterFixing;

		private Event(
				final double time,
				final double fixingWeightAtDate,
				final int decisionIndex,
				final double cumulativeWeightAfterFixing) {
			this.time = time;
			this.fixingWeightAtDate = fixingWeightAtDate;
			this.decisionIndex = decisionIndex;
			this.cumulativeWeightAfterFixing = cumulativeWeightAfterFixing;
		}

		private boolean hasFixing() {
			return fixingWeightAtDate > EPS;
		}

		private boolean hasDecision() {
			return decisionIndex >= 0;
		}
	}

	private static double[] fill(final int n, final double value) {
		final double[] x = new double[n];
		Arrays.fill(x, value);
		return x;
	}
}