/**
 * Script for running cell detection followed by trained object classifiers
 *
 * Note: before running the script in a new project, you need to manually
 * create a folder called "output" inside the QuPath project folder. Otherwise
 * the script will fail when exporting GeoJSON and CSV data.
 *
 * This script works also when running it in batchmode on a whole project.
 *
 * Author: Fredrik Nysjo
 */

import qupath.lib.objects.PathAnnotationObject;
import qupath.lib.objects.PathCellObject;
import qupath.lib.gui.tools.MeasurementExporter;
import qupath.lib.roi.ROIs;
import qupath.lib.regions.ImagePlane;

def imageData = getCurrentImageData();
def imageName = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName());
def outputName = imageName;
def outputDir = getProject().getPath().getParent().toString() + "/output/";

// Make sure channel names matches those in the trained classifiers
setChannelNames("DAPI", "Opal 480", "Opal 520", "Opal 540", "Opal 570",
                "Opal 620", "Opal 650", "Opal 690","Opal 780", "Autoflourescence")

// Find annotation for core
if (getAnnotationObjects().isEmpty()) {
    // Create an initial annotation by placing a fixed-size circle in the image
    def plane = ImagePlane.getPlane(0, 0);
    def roi = ROIs.createEllipseROI(800, 300, 2200, 2200, plane);
    def annotation = new PathAnnotationObject(roi);
    addObject(annotation);
    fireHierarchyUpdate();
}
selectAnnotations();

// Detect cells inside the annotation
clearDetections();
runPlugin('qupath.imagej.detect.cells.WatershedCellDetection', '{"detectionImage": "DAPI",  "requestedPixelSizeMicrons": 0.5,  "backgroundRadiusMicrons": 8.0,  "medianRadiusMicrons": 0.0,  "sigmaMicrons": 1.5,  "minAreaMicrons": 10.0,  "maxAreaMicrons": 400.0,  "threshold": 1.0,  "watershedPostProcess": true,  "cellExpansionMicrons": 2.5,  "includeNuclei": true,  "smoothBoundaries": true,  "makeMeasurements": true}');

// Apply classifiers sequentially on the cells
selectDetections();
runObjectClassifier("Leukocyte_tmi_P2", "Oligodendrocyte_tmi_P2", "PVC_tmi_P2", "Glioma cell_tmi_P2", "Endothelial cell_tmi_P2", "TAMM_tmi_P2", "Ki-67_tmi_P2", "Astrocyte_tmi_P2");

// Calculate distance and neighborhood measurements
/*detectionCentroidDistances(true);
selectAnnotations();
runPlugin('qupath.opencv.features.DelaunayClusteringPlugin', '{"distanceThresholdMicrons":0.0,"limitByClass":false,"addClusterMeasurements":false}');*/

// Save region data (as GeoJSON) and point data (as CSV)
exportSelectedObjectsToGeoJson(outputDir + outputName + ".json", "PRETTY_JSON", "FEATURE_COLLECTION")
saveDetectionMeasurements(outputDir + outputName + ".csv");
