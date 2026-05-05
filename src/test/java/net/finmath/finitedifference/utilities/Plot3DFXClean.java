package net.finmath.finitedifference.utilities;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.DoubleBinaryOperator;

import javax.imageio.IIOImage;
import javax.imageio.ImageIO;
import javax.imageio.ImageWriteParam;
import javax.imageio.ImageWriter;
import javax.imageio.stream.ImageOutputStream;
import javax.swing.JFrame;
import javax.swing.SwingUtilities;

import javafx.application.Platform;
import javafx.embed.swing.JFXPanel;
import javafx.embed.swing.SwingFXUtils;
import javafx.geometry.Insets;
import javafx.geometry.Point3D;
import javafx.geometry.Pos;
import javafx.scene.AmbientLight;
import javafx.scene.DepthTest;
import javafx.scene.Group;
import javafx.scene.Node;
import javafx.scene.PerspectiveCamera;
import javafx.scene.PointLight;
import javafx.scene.Scene;
import javafx.scene.SceneAntialiasing;
import javafx.scene.SubScene;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.image.PixelWriter;
import javafx.scene.image.WritableImage;
import javafx.scene.input.KeyCode;
import javafx.scene.input.MouseButton;
import javafx.scene.input.ScrollEvent;
import javafx.scene.input.ZoomEvent;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.StackPane;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.paint.PhongMaterial;
import javafx.scene.shape.CullFace;
import javafx.scene.shape.Cylinder;
import javafx.scene.shape.DrawMode;
import javafx.scene.shape.MeshView;
import javafx.scene.shape.TriangleMesh;
import javafx.scene.text.Text;
import javafx.scene.transform.Rotate;
import javafx.stage.FileChooser;
import net.finmath.plots.Named;
import net.finmath.plots.Plot;

/**
 * JavaFX 3D surface plot inspired by MATLAB's surf.
 *
 * Features:
 * - filled colored surface
 * - wireframe overlay
 * - white background
 * - axis labels and tick labels
 * - title
 * - optional color bar via setIsLegendVisible(true)
 * - automatic fitted initial view
 * - mouse drag rotation
 * - scroll and touchpad pinch zoom
 * - reset via double click, R, 0, or toolbar button
 * - export to JPG, PDF, SVG
 * - high-resolution export via setExportScale(...)
 */
public class Plot3DFXClean implements Plot {

    private static final int DEFAULT_WINDOW_WIDTH = 1000;
    private static final int DEFAULT_WINDOW_HEIGHT = 760;

    private static final double PLOT_WIDTH = 500.0;
    private static final double PLOT_DEPTH = 500.0;
    private static final double PLOT_HEIGHT = 320.0;

    private static final double CAMERA_FOV_DEGREES = 35.0;

    /*
     * Controls how much of the available SubScene is occupied by the fitted plot.
     * Increase to 0.90 or 0.93 for a larger initial plot.
     * Decrease to 0.80 for more white margin.
     */
    private static final double FIT_FRACTION = 0.88;

    private static final double DEFAULT_ROTATE_X = 25.0;
    private static final double DEFAULT_ROTATE_Y = -35.0;

    private static final double MIN_USER_SCALE = 0.20;
    private static final double MAX_USER_SCALE = 8.00;

    private static final double DEFAULT_EXPORT_SCALE = 3.0;
    private static final float JPEG_QUALITY = 0.98f;

    private static final int TICK_COUNT = 5;

    private static volatile boolean javaFXInitialized;

    private final double xmin;
    private final double xmax;
    private final double ymin;
    private final double ymax;
    private final int numberOfPointsX;
    private final int numberOfPointsY;
    private final Named<DoubleBinaryOperator> function;

    private String title = "";
    private String xAxisLabel = "x";
    private String yAxisLabel = "y";
    private String zAxisLabel = "z";
    private Boolean isLegendVisible = false;

    private double exportScale = DEFAULT_EXPORT_SCALE;

    private double zmin;
    private double zmax;

    private double mouseOldX;
    private double mouseOldY;

    private double rotateXAngle = DEFAULT_ROTATE_X;
    private double rotateYAngle = DEFAULT_ROTATE_Y;
    private double userScale = 1.0;

    private transient JFrame frame;
    private transient BorderPane root;
    private transient BorderPane chartRoot;

    private transient Group currentWorld;
    private transient Rotate currentRotateX;
    private transient Rotate currentRotateY;
    private transient PerspectiveCamera currentCamera;
    private transient SubScene currentSubScene;

    private final Object updateLock = new Object();

    public Plot3DFXClean(
            final double xmin,
            final double xmax,
            final double ymin,
            final double ymax,
            final int numberOfPointsX,
            final int numberOfPointsY,
            final Named<DoubleBinaryOperator> function) {

        if(numberOfPointsX < 2) {
            throw new IllegalArgumentException("numberOfPointsX must be at least 2.");
        }

        if(numberOfPointsY < 2) {
            throw new IllegalArgumentException("numberOfPointsY must be at least 2.");
        }

        if(function == null) {
            throw new IllegalArgumentException("function must not be null.");
        }

        this.xmin = xmin;
        this.xmax = xmax;
        this.ymin = ymin;
        this.ymax = ymax;
        this.numberOfPointsX = numberOfPointsX;
        this.numberOfPointsY = numberOfPointsY;
        this.function = function;
    }

    public Plot3DFXClean(
            final double xmin,
            final double xmax,
            final double ymin,
            final double ymax,
            final int numberOfPointsX,
            final int numberOfPointsY,
            final DoubleBinaryOperator function) {

        this(
                xmin,
                xmax,
                ymin,
                ymax,
                numberOfPointsX,
                numberOfPointsY,
                new Named<>("", function)
        );
    }

    @Override
    public void show() {
        ensureJavaFXInitialized();

        SwingUtilities.invokeLater(() -> {
            synchronized(updateLock) {
                if(frame != null) {
                    frame.dispose();
                    frame = null;
                }

                frame = new JFrame(getWindowTitle());
                frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);

                final JFXPanel fxPanel = new JFXPanel();
                fxPanel.setPreferredSize(new java.awt.Dimension(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT));

                frame.add(fxPanel);
                frame.pack();
                frame.setLocationRelativeTo(null);
                frame.setVisible(true);

                Platform.runLater(() -> {
                    root = createApplicationRoot();

                    final Scene scene = new Scene(root, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT);
                    scene.setFill(Color.WHITE);

                    installKeyboardHandlers(scene);

                    fxPanel.setScene(scene);
                    root.requestFocus();
                });
            }
        });
    }

    @Override
    public void close() {
        SwingUtilities.invokeLater(() -> {
            synchronized(updateLock) {
                if(frame != null) {
                    frame.dispose();
                    frame = null;
                }
            }
        });
    }

    private BorderPane createApplicationRoot() {
        final BorderPane applicationRoot = new BorderPane();

        applicationRoot.setStyle("-fx-background-color: white;");
        applicationRoot.setPadding(new Insets(8.0));
        applicationRoot.setFocusTraversable(true);

        chartRoot = createChartRoot(true, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT);

        applicationRoot.setTop(createToolbar());
        applicationRoot.setCenter(chartRoot);
        applicationRoot.setBottom(createHintLabel());

        return applicationRoot;
    }

    private BorderPane createChartRoot(
            final boolean interactive,
            final double width,
            final double height) {

        final double[][] values = evaluateFunction();

        final Group plotGroup = new Group();

        final MeshView surface = createSurface(values);
        final MeshView wireframe = createWireframe(surface);

        plotGroup.getChildren().addAll(
                createAxes(),
                surface,
                wireframe
        );

        final Rotate localRotateX = new Rotate(rotateXAngle, Rotate.X_AXIS);
        final Rotate localRotateY = new Rotate(rotateYAngle, Rotate.Y_AXIS);

        final Group rotatableWorld = new Group(plotGroup);
        rotatableWorld.getTransforms().addAll(localRotateX, localRotateY);
        applyUserScale(rotatableWorld);

        final AmbientLight ambientLight = new AmbientLight(Color.rgb(180, 180, 180));

        final PointLight pointLight = new PointLight(Color.WHITE);
        pointLight.setTranslateX(-450.0);
        pointLight.setTranslateY(-700.0);
        pointLight.setTranslateZ(-900.0);

        final Group sceneRoot3D = new Group(rotatableWorld, ambientLight, pointLight);

        final PerspectiveCamera camera = new PerspectiveCamera(true);
        camera.setFieldOfView(CAMERA_FOV_DEGREES);
        camera.setVerticalFieldOfView(true);
        camera.setNearClip(0.1);
        camera.setFarClip(20_000.0);

        final SubScene subScene = new SubScene(
                sceneRoot3D,
                width,
                height,
                true,
                SceneAntialiasing.BALANCED
        );

        subScene.setFill(Color.WHITE);
        subScene.setCamera(camera);

        final StackPane centerPane = new StackPane(subScene);
        centerPane.setStyle("-fx-background-color: white;");

        subScene.widthProperty().bind(centerPane.widthProperty());
        subScene.heightProperty().bind(centerPane.heightProperty());

        subScene.widthProperty().addListener((observable, oldValue, newValue) ->
                updateCameraDistance(camera, subScene.getWidth(), subScene.getHeight())
        );

        subScene.heightProperty().addListener((observable, oldValue, newValue) ->
                updateCameraDistance(camera, subScene.getWidth(), subScene.getHeight())
        );

        updateCameraDistance(camera, Math.max(width, 1.0), Math.max(height, 1.0));

        if(interactive) {
            currentWorld = rotatableWorld;
            currentRotateX = localRotateX;
            currentRotateY = localRotateY;
            currentCamera = camera;
            currentSubScene = subScene;

            installMouseHandlers(subScene);
            installZoomHandlers(subScene);
        }

        final BorderPane chart = new BorderPane();

        chart.setStyle("-fx-background-color: white;");
        chart.setPadding(new Insets(4.0));

        final String effectiveTitle = getEffectiveTitle();

        if(!effectiveTitle.isEmpty()) {
            final Label titleLabel = new Label(effectiveTitle);
            titleLabel.setStyle("-fx-font-size: 18px; -fx-font-weight: bold;");
            titleLabel.setPadding(new Insets(0.0, 0.0, 6.0, 0.0));

            BorderPane.setAlignment(titleLabel, Pos.CENTER);
            chart.setTop(titleLabel);
        }

        chart.setCenter(centerPane);

        if(Boolean.TRUE.equals(isLegendVisible)) {
            chart.setRight(createColorBar());
        }

        return chart;
    }

    private HBox createToolbar() {
        final Button resetButton = new Button("Reset view");
        resetButton.setOnAction(event -> resetView());

        final Button jpgButton = new Button("Export JPG");
        jpgButton.setOnAction(event -> exportWithFileChooser("JPG image", "*.jpg", "jpg"));

        final Button pdfButton = new Button("Export PDF");
        pdfButton.setOnAction(event -> exportWithFileChooser("PDF document", "*.pdf", "pdf"));

        final Button svgButton = new Button("Export SVG");
        svgButton.setOnAction(event -> exportWithFileChooser("SVG image", "*.svg", "svg"));

        final HBox toolbar = new HBox(8.0, resetButton, jpgButton, pdfButton, svgButton);

        toolbar.setAlignment(Pos.CENTER_LEFT);
        toolbar.setPadding(new Insets(0.0, 0.0, 6.0, 0.0));

        return toolbar;
    }

    private Label createHintLabel() {
        final Label hint = new Label(
                "Drag to rotate. Scroll or pinch to zoom. Double-click, press R, press 0, or use Reset view to return to the fitted view."
        );

        hint.setStyle("-fx-font-size: 11px; -fx-text-fill: #555555;");
        hint.setPadding(new Insets(6.0, 0.0, 0.0, 0.0));

        return hint;
    }

    private void installKeyboardHandlers(final Scene scene) {
        scene.setOnKeyPressed(event -> {
            if(event.getCode() == KeyCode.R
                    || event.getCode() == KeyCode.DIGIT0
                    || event.getCode() == KeyCode.NUMPAD0) {

                resetView();
                event.consume();

            } else if(event.getCode() == KeyCode.PLUS
                    || event.getCode() == KeyCode.EQUALS
                    || event.getCode() == KeyCode.ADD) {

                zoomBy(1.15);
                event.consume();

            } else if(event.getCode() == KeyCode.MINUS
                    || event.getCode() == KeyCode.SUBTRACT) {

                zoomBy(1.0 / 1.15);
                event.consume();
            }
        });
    }

    private void installMouseHandlers(final SubScene subScene) {
        subScene.setOnMousePressed(mouseEvent -> {
            mouseOldX = mouseEvent.getSceneX();
            mouseOldY = mouseEvent.getSceneY();

            if(root != null) {
                root.requestFocus();
            }

            if(mouseEvent.getButton() == MouseButton.PRIMARY && mouseEvent.getClickCount() == 2) {
                resetView();
                mouseEvent.consume();
            }
        });

        subScene.setOnMouseDragged(mouseEvent -> {
            final double mouseX = mouseEvent.getSceneX();
            final double mouseY = mouseEvent.getSceneY();

            rotateXAngle -= 0.7 * (mouseY - mouseOldY);
            rotateYAngle += 0.7 * (mouseX - mouseOldX);

            if(currentRotateX != null) {
                currentRotateX.setAngle(rotateXAngle);
            }

            if(currentRotateY != null) {
                currentRotateY.setAngle(rotateYAngle);
            }

            if(currentCamera != null && currentSubScene != null && userScale <= 1.0) {
                updateCameraDistance(currentCamera, currentSubScene.getWidth(), currentSubScene.getHeight());
            }

            mouseOldX = mouseX;
            mouseOldY = mouseY;

            mouseEvent.consume();
        });
    }

    private void installZoomHandlers(final SubScene subScene) {
        subScene.addEventFilter(ScrollEvent.SCROLL, event -> {
            final double factor = Math.pow(1.0025, event.getDeltaY());
            zoomBy(factor);
            event.consume();
        });

        subScene.addEventFilter(ZoomEvent.ZOOM, event -> {
            zoomBy(event.getZoomFactor());
            event.consume();
        });
    }

    private void resetView() {
        rotateXAngle = DEFAULT_ROTATE_X;
        rotateYAngle = DEFAULT_ROTATE_Y;
        userScale = 1.0;

        if(currentRotateX != null) {
            currentRotateX.setAngle(rotateXAngle);
        }

        if(currentRotateY != null) {
            currentRotateY.setAngle(rotateYAngle);
        }

        if(currentWorld != null) {
            applyUserScale(currentWorld);
        }

        if(currentCamera != null && currentSubScene != null) {
            updateCameraDistance(currentCamera, currentSubScene.getWidth(), currentSubScene.getHeight());
        }
    }

    private void zoomBy(final double factor) {
        if(!Double.isFinite(factor) || factor <= 0.0) {
            return;
        }

        userScale = clamp(userScale * factor, MIN_USER_SCALE, MAX_USER_SCALE);

        if(currentWorld != null) {
            applyUserScale(currentWorld);
        }
    }

    private void applyUserScale(final Node node) {
        node.setScaleX(userScale);
        node.setScaleY(userScale);
        node.setScaleZ(userScale);
    }

    private void updateCameraDistance(
            final PerspectiveCamera camera,
            final double viewportWidth,
            final double viewportHeight) {

        final double safeWidth = Math.max(viewportWidth, 1.0);
        final double safeHeight = Math.max(viewportHeight, 1.0);
        final double aspectRatio = safeWidth / safeHeight;

        final double verticalHalfFov = Math.toRadians(CAMERA_FOV_DEGREES / 2.0);
        final double horizontalHalfFov = Math.atan(Math.tan(verticalHalfFov) * aspectRatio);

        final double tanVertical = Math.tan(verticalHalfFov);
        final double tanHorizontal = Math.tan(horizontalHalfFov);

        double requiredDistance = 1.0;

        for(final Point3D point : createFitBoundingBoxCorners()) {
            final Point3D rotatedPoint = rotateForCurrentView(point);

            final double xRequirement =
                    Math.abs(rotatedPoint.getX()) / (tanHorizontal * FIT_FRACTION)
                            - rotatedPoint.getZ();

            final double yRequirement =
                    Math.abs(rotatedPoint.getY()) / (tanVertical * FIT_FRACTION)
                            - rotatedPoint.getZ();

            requiredDistance = Math.max(requiredDistance, xRequirement);
            requiredDistance = Math.max(requiredDistance, yRequirement);
        }

        camera.setTranslateZ(-requiredDistance);
    }

    private List<Point3D> createFitBoundingBoxCorners() {
        final double xMin = -PLOT_WIDTH / 2.0 - 110.0;
        final double xMax =  PLOT_WIDTH / 2.0 + 45.0;

        final double yMin = -PLOT_HEIGHT / 2.0 - 30.0;
        final double yMax =  PLOT_HEIGHT / 2.0 + 85.0;

        final double zMin = -PLOT_DEPTH / 2.0 - 55.0;
        final double zMax =  PLOT_DEPTH / 2.0 + 45.0;

        final List<Point3D> corners = new ArrayList<>();

        corners.add(new Point3D(xMin, yMin, zMin));
        corners.add(new Point3D(xMin, yMin, zMax));
        corners.add(new Point3D(xMin, yMax, zMin));
        corners.add(new Point3D(xMin, yMax, zMax));

        corners.add(new Point3D(xMax, yMin, zMin));
        corners.add(new Point3D(xMax, yMin, zMax));
        corners.add(new Point3D(xMax, yMax, zMin));
        corners.add(new Point3D(xMax, yMax, zMax));

        return corners;
    }

    private Point3D rotateForCurrentView(final Point3D point) {
        final Point3D afterRotateX = rotateAroundX(point, Math.toRadians(rotateXAngle));
        final Point3D afterRotateY = rotateAroundY(afterRotateX, Math.toRadians(rotateYAngle));

        return new Point3D(
                afterRotateY.getX() * userScale,
                afterRotateY.getY() * userScale,
                afterRotateY.getZ() * userScale
        );
    }

    private Point3D rotateAroundX(final Point3D point, final double angle) {
        final double cos = Math.cos(angle);
        final double sin = Math.sin(angle);

        final double y = point.getY() * cos - point.getZ() * sin;
        final double z = point.getY() * sin + point.getZ() * cos;

        return new Point3D(point.getX(), y, z);
    }

    private Point3D rotateAroundY(final Point3D point, final double angle) {
        final double cos = Math.cos(angle);
        final double sin = Math.sin(angle);

        final double x = point.getX() * cos + point.getZ() * sin;
        final double z = -point.getX() * sin + point.getZ() * cos;

        return new Point3D(x, point.getY(), z);
    }

    private double[][] evaluateFunction() {
        final double[][] values = new double[numberOfPointsX][numberOfPointsY];

        zmin = Double.POSITIVE_INFINITY;
        zmax = Double.NEGATIVE_INFINITY;

        for(int i = 0; i < numberOfPointsX; i++) {
            final double x = xmin + i * (xmax - xmin) / (numberOfPointsX - 1.0);

            for(int j = 0; j < numberOfPointsY; j++) {
                final double y = ymin + j * (ymax - ymin) / (numberOfPointsY - 1.0);

                double value;

                try {
                    value = function.get().applyAsDouble(x, y);
                } catch(final RuntimeException exception) {
                    value = Double.NaN;
                }

                values[i][j] = value;

                if(Double.isFinite(value)) {
                    zmin = Math.min(zmin, value);
                    zmax = Math.max(zmax, value);
                }
            }
        }

        if(!Double.isFinite(zmin) || !Double.isFinite(zmax)) {
            zmin = 0.0;
            zmax = 1.0;
        }

        if(Math.abs(zmax - zmin) < 1E-14) {
            final double center = zmin;
            zmin = center - 0.5;
            zmax = center + 0.5;
        }

        return values;
    }

    private MeshView createSurface(final double[][] values) {
        final TriangleMesh mesh = new TriangleMesh();

        for(int i = 0; i < numberOfPointsX; i++) {
            final double tx = i / (numberOfPointsX - 1.0);
            final float x = (float) (-PLOT_WIDTH / 2.0 + tx * PLOT_WIDTH);

            for(int j = 0; j < numberOfPointsY; j++) {
                final double ty = j / (numberOfPointsY - 1.0);
                final float z = (float) (-PLOT_DEPTH / 2.0 + ty * PLOT_DEPTH);

                final double value = values[i][j];
                final float y = (float) mapValueToDisplayY(value);

                mesh.getPoints().addAll(x, y, z);
                mesh.getTexCoords().addAll((float) tx, (float) ty);
            }
        }

        for(int i = 0; i < numberOfPointsX - 1; i++) {
            for(int j = 0; j < numberOfPointsY - 1; j++) {
                final int p00 = i * numberOfPointsY + j;
                final int p01 = i * numberOfPointsY + j + 1;
                final int p10 = (i + 1) * numberOfPointsY + j;
                final int p11 = (i + 1) * numberOfPointsY + j + 1;

                mesh.getFaces().addAll(
                        p00, p00, p01, p01, p10, p10,
                        p10, p10, p01, p01, p11, p11
                );

                mesh.getFaceSmoothingGroups().addAll(1, 1);
            }
        }

        final PhongMaterial material = new PhongMaterial();

        material.setDiffuseMap(createColorMap(values));
        material.setSpecularColor(Color.rgb(60, 60, 60));
        material.setSpecularPower(12.0);

        final MeshView surface = new MeshView(mesh);

        surface.setMaterial(material);
        surface.setCullFace(CullFace.NONE);
        surface.setDrawMode(DrawMode.FILL);
        surface.setDepthTest(DepthTest.ENABLE);

        return surface;
    }

    private MeshView createWireframe(final MeshView surface) {
        final MeshView wireframe = new MeshView(surface.getMesh());
        final PhongMaterial material = new PhongMaterial(Color.rgb(20, 20, 20, 0.45));

        wireframe.setMaterial(material);
        wireframe.setCullFace(CullFace.NONE);
        wireframe.setDrawMode(DrawMode.LINE);
        wireframe.setDepthTest(DepthTest.ENABLE);
        wireframe.setMouseTransparent(true);

        return wireframe;
    }

    private Group createAxes() {
        final Group axes = new Group();

        final double x0 = -PLOT_WIDTH / 2.0;
        final double x1 = PLOT_WIDTH / 2.0;

        final double yBottom = PLOT_HEIGHT / 2.0;
        final double yTop = -PLOT_HEIGHT / 2.0;

        final double z0 = -PLOT_DEPTH / 2.0;
        final double z1 = PLOT_DEPTH / 2.0;

        final Color axisColor = Color.BLACK;
        final Color gridColor = Color.rgb(175, 175, 175, 0.65);

        addBoxEdges(axes, x0, x1, yBottom, yTop, z0, z1, axisColor);

        for(int k = 0; k <= TICK_COUNT; k++) {
            final double t = k / (double) TICK_COUNT;

            final double x = x0 + t * PLOT_WIDTH;
            final double z = z0 + t * PLOT_DEPTH;
            final double y = yBottom - t * PLOT_HEIGHT;

            axes.getChildren().add(segment(x, yBottom, z0, x, yBottom, z1, 0.35, gridColor));
            axes.getChildren().add(segment(x0, yBottom, z, x1, yBottom, z, 0.35, gridColor));

            axes.getChildren().add(segment(x, yBottom, z1, x, yTop, z1, 0.35, gridColor));
            axes.getChildren().add(segment(x0, y, z1, x1, y, z1, 0.35, gridColor));

            axes.getChildren().add(segment(x0, yBottom, z, x0, yTop, z, 0.35, gridColor));
            axes.getChildren().add(segment(x0, y, z0, x0, y, z1, 0.35, gridColor));

            axes.getChildren().add(text3D(
                    format(xmin + t * (xmax - xmin)),
                    x - 14.0,
                    yBottom + 24.0,
                    z0 - 8.0,
                    11,
                    false
            ));

            axes.getChildren().add(text3D(
                    format(ymin + t * (ymax - ymin)),
                    x0 - 58.0,
                    yBottom + 14.0,
                    z - 4.0,
                    11,
                    false
            ));

            axes.getChildren().add(text3D(
                    format(zmin + t * (zmax - zmin)),
                    x0 - 72.0,
                    y + 4.0,
                    z0 - 4.0,
                    11,
                    false
            ));
        }

        axes.getChildren().add(text3D(
                nullToEmpty(xAxisLabel),
                -20.0,
                yBottom + 58.0,
                z0 - 22.0,
                14,
                true
        ));

        axes.getChildren().add(text3D(
                nullToEmpty(yAxisLabel),
                x0 - 78.0,
                yBottom + 44.0,
                0.0,
                14,
                true
        ));

        axes.getChildren().add(text3D(
                nullToEmpty(zAxisLabel),
                x0 - 98.0,
                0.0,
                z0 - 18.0,
                14,
                true
        ));

        return axes;
    }

    private void addBoxEdges(
            final Group group,
            final double x0,
            final double x1,
            final double yBottom,
            final double yTop,
            final double z0,
            final double z1,
            final Color color) {

        final double r = 0.85;

        group.getChildren().add(segment(x0, yBottom, z0, x1, yBottom, z0, r, color));
        group.getChildren().add(segment(x1, yBottom, z0, x1, yBottom, z1, r, color));
        group.getChildren().add(segment(x1, yBottom, z1, x0, yBottom, z1, r, color));
        group.getChildren().add(segment(x0, yBottom, z1, x0, yBottom, z0, r, color));

        group.getChildren().add(segment(x0, yTop, z0, x1, yTop, z0, r, color));
        group.getChildren().add(segment(x1, yTop, z0, x1, yTop, z1, r, color));
        group.getChildren().add(segment(x1, yTop, z1, x0, yTop, z1, r, color));
        group.getChildren().add(segment(x0, yTop, z1, x0, yTop, z0, r, color));

        group.getChildren().add(segment(x0, yBottom, z0, x0, yTop, z0, r, color));
        group.getChildren().add(segment(x1, yBottom, z0, x1, yTop, z0, r, color));
        group.getChildren().add(segment(x1, yBottom, z1, x1, yTop, z1, r, color));
        group.getChildren().add(segment(x0, yBottom, z1, x0, yTop, z1, r, color));
    }

    private Node segment(
            final double xStart,
            final double yStart,
            final double zStart,
            final double xEnd,
            final double yEnd,
            final double zEnd,
            final double radius,
            final Color color) {

        final Point3D start = new Point3D(xStart, yStart, zStart);
        final Point3D end = new Point3D(xEnd, yEnd, zEnd);
        final Point3D diff = end.subtract(start);

        final double length = diff.magnitude();

        if(length < 1E-12) {
            return new Group();
        }

        final Cylinder cylinder = new Cylinder(radius, length);

        cylinder.setMaterial(new PhongMaterial(color));
        cylinder.setDepthTest(DepthTest.ENABLE);

        final Point3D midpoint = start.midpoint(end);

        cylinder.setTranslateX(midpoint.getX());
        cylinder.setTranslateY(midpoint.getY());
        cylinder.setTranslateZ(midpoint.getZ());

        final Point3D yAxis = new Point3D(0.0, 1.0, 0.0);
        final Point3D rotationAxis = yAxis.crossProduct(diff);

        if(rotationAxis.magnitude() > 1E-12) {
            final double angle = Math.toDegrees(Math.acos(
                    clamp(yAxis.normalize().dotProduct(diff.normalize()), -1.0, 1.0)
            ));

            cylinder.getTransforms().add(new Rotate(angle, rotationAxis));
        }

        return cylinder;
    }

    private Text text3D(
            final String value,
            final double x,
            final double y,
            final double z,
            final int fontSize,
            final boolean bold) {

        final Text text = new Text(value == null ? "" : value);

        text.setFill(Color.BLACK);
        text.setDepthTest(DepthTest.DISABLE);
        text.setMouseTransparent(true);

        text.setTranslateX(x);
        text.setTranslateY(y);
        text.setTranslateZ(z);

        text.setStyle(
                "-fx-font-size: " + fontSize + "px;"
                        + (bold ? "-fx-font-weight: bold;" : "")
        );

        return text;
    }

    private Image createColorMap(final double[][] values) {
        final WritableImage image = new WritableImage(numberOfPointsX, numberOfPointsY);
        final PixelWriter writer = image.getPixelWriter();

        final double range = zmax - zmin;

        for(int i = 0; i < numberOfPointsX; i++) {
            for(int j = 0; j < numberOfPointsY; j++) {
                final double value = values[i][j];

                if(!Double.isFinite(value)) {
                    writer.setColor(i, j, Color.LIGHTGRAY);
                    continue;
                }

                final double t = clamp((value - zmin) / range, 0.0, 1.0);
                writer.setColor(i, j, matlabLikeColor(t));
            }
        }

        return image;
    }

    private VBox createColorBar() {
        final VBox box = new VBox(5.0);

        box.setAlignment(Pos.CENTER);
        box.setPadding(new Insets(25.0, 10.0, 25.0, 14.0));

        final Label label = new Label(nullToEmpty(zAxisLabel));
        label.setStyle("-fx-font-size: 12px; -fx-font-weight: bold;");

        final Label maxLabel = new Label(format(zmax));
        final Label minLabel = new Label(format(zmin));

        final ImageView imageView = new ImageView(createColorBarImage(24, 260));
        imageView.setFitWidth(24.0);
        imageView.setFitHeight(260.0);

        box.getChildren().addAll(label, maxLabel, imageView, minLabel);

        return box;
    }

    private Image createColorBarImage(final int width, final int height) {
        final WritableImage image = new WritableImage(width, height);
        final PixelWriter writer = image.getPixelWriter();

        for(int y = 0; y < height; y++) {
            final double t = 1.0 - y / (height - 1.0);
            final Color color = matlabLikeColor(t);

            for(int x = 0; x < width; x++) {
                writer.setColor(x, y, color);
            }
        }

        return image;
    }

    private Color matlabLikeColor(final double value) {
        final double t = clamp(value, 0.0, 1.0);

        if(t < 0.20) {
            return Color.rgb(0, 0, 130).interpolate(Color.BLUE, t / 0.20);
        }

        if(t < 0.40) {
            return Color.BLUE.interpolate(Color.CYAN, (t - 0.20) / 0.20);
        }

        if(t < 0.60) {
            return Color.CYAN.interpolate(Color.LIMEGREEN, (t - 0.40) / 0.20);
        }

        if(t < 0.80) {
            return Color.LIMEGREEN.interpolate(Color.YELLOW, (t - 0.60) / 0.20);
        }

        return Color.YELLOW.interpolate(Color.RED, (t - 0.80) / 0.20);
    }

    private double mapValueToDisplayY(final double value) {
        final double finiteValue = Double.isFinite(value) ? value : zmin;
        final double t = clamp((finiteValue - zmin) / (zmax - zmin), 0.0, 1.0);

        return PLOT_HEIGHT / 2.0 - t * PLOT_HEIGHT;
    }

    private void exportWithFileChooser(
            final String description,
            final String pattern,
            final String extension) {

        final FileChooser fileChooser = new FileChooser();

        fileChooser.setTitle("Export " + extension.toUpperCase(Locale.US));
        fileChooser.getExtensionFilters().add(new FileChooser.ExtensionFilter(description, pattern));
        fileChooser.setInitialFileName(defaultExportFileName(extension));

        final File selectedFile = fileChooser.showSaveDialog(
                root == null ? null : root.getScene().getWindow()
        );

        if(selectedFile == null) {
            return;
        }

        final File file = ensureExtension(selectedFile, extension);

        final int logicalWidth = chartRoot == null || chartRoot.getWidth() <= 0.0
                ? DEFAULT_WINDOW_WIDTH
                : (int) Math.round(chartRoot.getWidth());

        final int logicalHeight = chartRoot == null || chartRoot.getHeight() <= 0.0
                ? DEFAULT_WINDOW_HEIGHT
                : (int) Math.round(chartRoot.getHeight());

        final int exportWidth = Math.max(1, (int) Math.round(logicalWidth * exportScale));
        final int exportHeight = Math.max(1, (int) Math.round(logicalHeight * exportScale));

        try {
            if("jpg".equalsIgnoreCase(extension) || "jpeg".equalsIgnoreCase(extension)) {
                saveAsJPG(file, exportWidth, exportHeight);
            } else if("pdf".equalsIgnoreCase(extension)) {
                saveAsPDF(file, logicalWidth, logicalHeight);
            } else if("svg".equalsIgnoreCase(extension)) {
                saveAsSVG(file, exportWidth, exportHeight);
            }
        } catch(final RuntimeException | IOException exception) {
            showExportError(exception);
        }
    }

    private String defaultExportFileName(final String extension) {
        final String baseName = getEffectiveTitle().isEmpty()
                ? "surface-plot"
                : sanitizeFileName(getEffectiveTitle());

        return baseName + "." + extension;
    }

    private String sanitizeFileName(final String value) {
        return value.replaceAll("[^a-zA-Z0-9._-]+", "-");
    }

    private File ensureExtension(final File file, final String extension) {
        final String lowerCaseName = file.getName().toLowerCase(Locale.US);
        final String normalizedExtension = "." + extension.toLowerCase(Locale.US);

        if(lowerCaseName.endsWith(normalizedExtension)) {
            return file;
        }

        return new File(file.getParentFile(), file.getName() + normalizedExtension);
    }

    private void showExportError(final Throwable throwable) {
        final Alert alert = new Alert(Alert.AlertType.ERROR);

        alert.setTitle("Export failed");
        alert.setHeaderText("The plot could not be exported.");
        alert.setContentText(
                throwable.getMessage() == null
                        ? throwable.toString()
                        : throwable.getMessage()
        );

        alert.showAndWait();
    }

    @Override
    public Plot saveAsJPG(final File file, final int width, final int height) throws IOException {
        final WritableImage image = createSnapshot(width, height);
        final BufferedImage bufferedImage = toRgbImage(SwingFXUtils.fromFXImage(image, null));

        ensureParentDirectoryExists(file);
        writeJpeg(bufferedImage, file, JPEG_QUALITY);

        return this;
    }

    @Override
    public Plot saveAsPDF(final File file, final int width, final int height) {
        try {
            final int pageWidth = width > 0 ? width : DEFAULT_WINDOW_WIDTH;
            final int pageHeight = height > 0 ? height : DEFAULT_WINDOW_HEIGHT;

            final int renderWidth = Math.max(1, (int) Math.round(pageWidth * exportScale));
            final int renderHeight = Math.max(1, (int) Math.round(pageHeight * exportScale));

            final WritableImage image = createSnapshot(renderWidth, renderHeight);
            final BufferedImage bufferedImage = toRgbImage(SwingFXUtils.fromFXImage(image, null));

            ensureParentDirectoryExists(file);
            writeImagePdf(file, bufferedImage, pageWidth, pageHeight);

            return this;
        } catch(final IOException exception) {
            throw new UncheckedIOException(exception);
        }
    }

    @Override
    public Plot saveAsSVG(final File file, final int width, final int height) {
        try {
            final WritableImage image = createSnapshot(width, height);
            final BufferedImage bufferedImage = SwingFXUtils.fromFXImage(image, null);

            final ByteArrayOutputStream pngBuffer = new ByteArrayOutputStream();

            if(!ImageIO.write(bufferedImage, "png", pngBuffer)) {
                throw new IOException("No PNG writer available.");
            }

            final String encodedPng = Base64.getEncoder().encodeToString(pngBuffer.toByteArray());

            final String svg = ""
                    + "<svg xmlns=\"http://www.w3.org/2000/svg\" "
                    + "width=\"" + width + "\" height=\"" + height + "\" "
                    + "viewBox=\"0 0 " + width + " " + height + "\">\n"
                    + "<image x=\"0\" y=\"0\" width=\"" + width + "\" height=\"" + height + "\" "
                    + "href=\"data:image/png;base64," + encodedPng + "\"/>\n"
                    + "</svg>\n";

            ensureParentDirectoryExists(file);

            try(FileOutputStream outputStream = new FileOutputStream(file)) {
                outputStream.write(svg.getBytes(StandardCharsets.UTF_8));
            }

            return this;
        } catch(final IOException exception) {
            throw new UncheckedIOException(exception);
        }
    }

    private WritableImage createSnapshot(final int requestedWidth, final int requestedHeight) throws IOException {
        ensureJavaFXInitialized();

        final int width = requestedWidth > 0 ? requestedWidth : DEFAULT_WINDOW_WIDTH;
        final int height = requestedHeight > 0 ? requestedHeight : DEFAULT_WINDOW_HEIGHT;

        if(Platform.isFxApplicationThread()) {
            return createSnapshotOnFXThread(width, height);
        }

        final CountDownLatch latch = new CountDownLatch(1);
        final AtomicReference<WritableImage> imageReference = new AtomicReference<>();
        final AtomicReference<Throwable> errorReference = new AtomicReference<>();

        Platform.runLater(() -> {
            try {
                imageReference.set(createSnapshotOnFXThread(width, height));
            } catch(final Throwable throwable) {
                errorReference.set(throwable);
            } finally {
                latch.countDown();
            }
        });

        try {
            latch.await();
        } catch(final InterruptedException exception) {
            Thread.currentThread().interrupt();
            throw new IOException("Interrupted while creating plot snapshot.", exception);
        }

        if(errorReference.get() != null) {
            final Throwable throwable = errorReference.get();

            if(throwable instanceof IOException) {
                throw (IOException) throwable;
            }

            throw new IOException("Could not create plot snapshot.", throwable);
        }

        return imageReference.get();
    }

    private WritableImage createSnapshotOnFXThread(final int width, final int height) {
        final BorderPane exportRoot = createChartRoot(false, width, height);

        final Scene scene = new Scene(exportRoot, width, height);
        scene.setFill(Color.WHITE);

        exportRoot.resize(width, height);
        exportRoot.applyCss();
        exportRoot.layout();

        final WritableImage image = new WritableImage(width, height);
        scene.snapshot(image);

        return image;
    }

    private BufferedImage toRgbImage(final BufferedImage source) {
        final BufferedImage rgbImage = new BufferedImage(
                source.getWidth(),
                source.getHeight(),
                BufferedImage.TYPE_INT_RGB
        );

        final Graphics2D graphics = rgbImage.createGraphics();

        graphics.setColor(java.awt.Color.WHITE);
        graphics.fillRect(0, 0, source.getWidth(), source.getHeight());
        graphics.drawImage(source, 0, 0, null);
        graphics.dispose();

        return rgbImage;
    }

    private void writeImagePdf(
            final File file,
            final BufferedImage image,
            final double pageWidth,
            final double pageHeight) throws IOException {

        final byte[] jpegBytes = encodeJpeg(image, JPEG_QUALITY);

        final int imageWidth = image.getWidth();
        final int imageHeight = image.getHeight();

        final ByteArrayOutputStream pdf = new ByteArrayOutputStream();
        final List<Integer> offsets = new ArrayList<>();

        writeAscii(pdf, "%PDF-1.4\n");
        writeAscii(pdf, "%\u00e2\u00e3\u00cf\u00d3\n");

        offsets.add(pdf.size());
        writeAscii(pdf, "1 0 obj\n");
        writeAscii(pdf, "<< /Type /Catalog /Pages 2 0 R >>\n");
        writeAscii(pdf, "endobj\n");

        offsets.add(pdf.size());
        writeAscii(pdf, "2 0 obj\n");
        writeAscii(pdf, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>\n");
        writeAscii(pdf, "endobj\n");

        offsets.add(pdf.size());
        writeAscii(pdf, "3 0 obj\n");
        writeAscii(pdf, "<< /Type /Page /Parent 2 0 R ");
        writeAscii(pdf, "/MediaBox [0 0 " + formatPdfNumber(pageWidth) + " " + formatPdfNumber(pageHeight) + "] ");
        writeAscii(pdf, "/Resources << /XObject << /Im0 4 0 R >> >> ");
        writeAscii(pdf, "/Contents 5 0 R >>\n");
        writeAscii(pdf, "endobj\n");

        offsets.add(pdf.size());
        writeAscii(pdf, "4 0 obj\n");
        writeAscii(pdf, "<< /Type /XObject /Subtype /Image ");
        writeAscii(pdf, "/Width " + imageWidth + " ");
        writeAscii(pdf, "/Height " + imageHeight + " ");
        writeAscii(pdf, "/ColorSpace /DeviceRGB ");
        writeAscii(pdf, "/BitsPerComponent 8 ");
        writeAscii(pdf, "/Filter /DCTDecode ");
        writeAscii(pdf, "/Length " + jpegBytes.length + " >>\n");
        writeAscii(pdf, "stream\n");
        pdf.write(jpegBytes);
        writeAscii(pdf, "\nendstream\n");
        writeAscii(pdf, "endobj\n");

        final String contentStream = ""
                + "q\n"
                + formatPdfNumber(pageWidth) + " 0 0 " + formatPdfNumber(pageHeight) + " 0 0 cm\n"
                + "/Im0 Do\n"
                + "Q\n";

        final byte[] contentBytes = contentStream.getBytes(StandardCharsets.US_ASCII);

        offsets.add(pdf.size());
        writeAscii(pdf, "5 0 obj\n");
        writeAscii(pdf, "<< /Length " + contentBytes.length + " >>\n");
        writeAscii(pdf, "stream\n");
        pdf.write(contentBytes);
        writeAscii(pdf, "endstream\n");
        writeAscii(pdf, "endobj\n");

        final int xrefOffset = pdf.size();

        writeAscii(pdf, "xref\n");
        writeAscii(pdf, "0 6\n");
        writeAscii(pdf, "0000000000 65535 f \n");

        for(final int offset : offsets) {
            writeAscii(pdf, String.format(Locale.US, "%010d 00000 n \n", offset));
        }

        writeAscii(pdf, "trailer\n");
        writeAscii(pdf, "<< /Size 6 /Root 1 0 R >>\n");
        writeAscii(pdf, "startxref\n");
        writeAscii(pdf, Integer.toString(xrefOffset));
        writeAscii(pdf, "\n%%EOF\n");

        try(FileOutputStream outputStream = new FileOutputStream(file)) {
            pdf.writeTo(outputStream);
        }
    }

    private byte[] encodeJpeg(final BufferedImage image, final float quality) throws IOException {
        final ByteArrayOutputStream outputStream = new ByteArrayOutputStream();

        final Iterator<ImageWriter> writers = ImageIO.getImageWritersByFormatName("jpg");

        if(!writers.hasNext()) {
            throw new IOException("No JPG writer available.");
        }

        final ImageWriter writer = writers.next();

        try(ImageOutputStream imageOutputStream = ImageIO.createImageOutputStream(outputStream)) {
            writer.setOutput(imageOutputStream);

            final ImageWriteParam parameters = writer.getDefaultWriteParam();

            if(parameters.canWriteCompressed()) {
                parameters.setCompressionMode(ImageWriteParam.MODE_EXPLICIT);
                parameters.setCompressionQuality((float) clamp(quality, 0.01, 1.0));
            }

            writer.write(null, new IIOImage(image, null, null), parameters);
        } finally {
            writer.dispose();
        }

        return outputStream.toByteArray();
    }

    private void writeJpeg(
            final BufferedImage image,
            final File file,
            final float quality) throws IOException {

        final Iterator<ImageWriter> writers = ImageIO.getImageWritersByFormatName("jpg");

        if(!writers.hasNext()) {
            throw new IOException("No JPG writer available.");
        }

        final ImageWriter writer = writers.next();

        try(ImageOutputStream imageOutputStream = ImageIO.createImageOutputStream(file)) {
            writer.setOutput(imageOutputStream);

            final ImageWriteParam parameters = writer.getDefaultWriteParam();

            if(parameters.canWriteCompressed()) {
                parameters.setCompressionMode(ImageWriteParam.MODE_EXPLICIT);
                parameters.setCompressionQuality((float) clamp(quality, 0.01, 1.0));
            }

            writer.write(null, new IIOImage(image, null, null), parameters);
        } finally {
            writer.dispose();
        }
    }

    private void writeAscii(final ByteArrayOutputStream outputStream, final String text) throws IOException {
        outputStream.write(text.getBytes(StandardCharsets.ISO_8859_1));
    }

    private String formatPdfNumber(final double value) {
        return String.format(Locale.US, "%.4f", value)
                .replaceAll("0+$", "")
                .replaceAll("\\.$", "");
    }

    private void ensureParentDirectoryExists(final File file) throws IOException {
        final File parent = file.getAbsoluteFile().getParentFile();

        if(parent != null && !parent.exists() && !parent.mkdirs()) {
            throw new IOException("Could not create directory " + parent);
        }
    }

    public Image createImage(final double size, final float[][] noise) {
        final int width = (int) size;
        final int height = (int) size;

        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;

        for(int x = 0; x < width; x++) {
            for(int y = 0; y < height; y++) {
                final float value = noise[x][y];

                if(Float.isFinite(value)) {
                    min = Math.min(min, value);
                    max = Math.max(max, value);
                }
            }
        }

        if(!Double.isFinite(min) || !Double.isFinite(max) || Math.abs(max - min) < 1E-14) {
            min = -1.0;
            max = 1.0;
        }

        final WritableImage image = new WritableImage(width, height);
        final PixelWriter writer = image.getPixelWriter();

        for(int x = 0; x < width; x++) {
            for(int y = 0; y < height; y++) {
                final float value = noise[x][y];

                if(!Float.isFinite(value)) {
                    writer.setColor(x, y, Color.LIGHTGRAY);
                    continue;
                }

                final double t = clamp((value - min) / (max - min), 0.0, 1.0);
                writer.setColor(x, y, matlabLikeColor(t));
            }
        }

        return image;
    }

    public static double normalizeValue(
            final double value,
            final double min,
            final double max,
            final double newMin,
            final double newMax) {

        if(Math.abs(max - min) < 1E-14) {
            return 0.5 * (newMin + newMax);
        }

        return (value - min) * (newMax - newMin) / (max - min) + newMin;
    }

    public static double clamp(final double value, final double min, final double max) {
        if(Double.compare(value, min) < 0) {
            return min;
        }

        if(Double.compare(value, max) > 0) {
            return max;
        }

        return value;
    }

    private String getEffectiveTitle() {
        return title == null ? "" : title.trim();
    }

    private String getWindowTitle() {
        final String effectiveTitle = getEffectiveTitle();
        return effectiveTitle.isEmpty() ? "FX Surface Plot" : effectiveTitle;
    }

    private String nullToEmpty(final String value) {
        return value == null ? "" : value;
    }

    private String format(final double value) {
        if(Math.abs(value) < 1E-12) {
            return "0";
        }

        if(Math.abs(value) >= 1E4 || Math.abs(value) < 1E-3) {
            return String.format(Locale.US, "%.2e", value);
        }

        return String.format(Locale.US, "%.3f", value)
                .replaceAll("0+$", "")
                .replaceAll("\\.$", "");
    }

    private static synchronized void ensureJavaFXInitialized() {
        if(!javaFXInitialized) {
            new JFXPanel();
            javaFXInitialized = true;
        }
    }

    public Plot3DFXClean setExportScale(final double exportScale) {
        if(!Double.isFinite(exportScale) || exportScale <= 0.0) {
            throw new IllegalArgumentException("exportScale must be positive.");
        }

        this.exportScale = exportScale;
        return this;
    }

    public double getExportScale() {
        return exportScale;
    }

    @Override
    public Plot setTitle(final String title) {
        this.title = title;
        return this;
    }

    @Override
    public Plot setXAxisLabel(final String xAxisLabel) {
        this.xAxisLabel = xAxisLabel;
        return this;
    }

    @Override
    public Plot setYAxisLabel(final String yAxisLabel) {
        this.yAxisLabel = yAxisLabel;
        return this;
    }

    @Override
    public Plot setZAxisLabel(final String zAxisLabel) {
        this.zAxisLabel = zAxisLabel;
        return this;
    }

    @Override
    public Plot setIsLegendVisible(final Boolean isLegendVisible) {
        this.isLegendVisible = isLegendVisible;
        return this;
    }

    @Override
    public String toString() {
        return "Plot3DFXClean [xmin=" + xmin
                + ", xmax=" + xmax
                + ", ymin=" + ymin
                + ", ymax=" + ymax
                + ", numberOfPointsX=" + numberOfPointsX
                + ", numberOfPointsY=" + numberOfPointsY
                + ", function=" + function
                + ", title=" + title
                + ", xAxisLabel=" + xAxisLabel
                + ", yAxisLabel=" + yAxisLabel
                + ", zAxisLabel=" + zAxisLabel
                + ", isLegendVisible=" + isLegendVisible
                + ", exportScale=" + exportScale
                + "]";
    }
}