/**
 * Script for resolving ambiguous cells after classification.
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

numClasses = 7;
classes = ["Astrocyte", "Glioma cell", "TAMM", "Endothelial cell", "Leukocyte", "Oligodendrocyte", "PVC"];
measurements = ["Cell: Opal 780 max", "Cell: Opal 650 max", "Cell: Opal 480 max",
                "Cell: Opal 690 max", "Cell: Opal 540 max", "Cell: Opal 570 max",
                "Cell: Opal 620 max"];
maxValues = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
outputClasses = ["Astrocyte", "Glioma", "TAMM", "Endothelial", "Leukocyte", "Oligodendrocyte", "PVC", "Negative"];
outputColors = [0x2953b9, 0xff7f0e, 0x9467bd, 0x2699b1, 0xcc1111, 0x11cc11, 0x1111cc, 0x228b22];

// Get maximum feature value per cell type from non-ambiguous cells
for (cell in getDetectionObjects()) {
    def count = 0;  // Total number of classes in the cell
    for (int i = 0; i < numClasses; i++) {
        count += PathClassTools.containsName(cell.getPathClass(), classes[i]) ? 1 : 0;
    }

    for (int i = 0; i < numClasses; i++) {
        if (count == 1 && PathClassTools.containsName(cell.getPathClass(), classes[i])) {
            def value = cell.getMeasurementList().getMeasurementValue(measurements[i]);
            if (value > maxValues[i]) {
                maxValues[i] = value;
            }
        }
    }
}

// Find the "closest" cell type for each cell to resolve ambiguities.
// In addition, cells that are marker negative will be given a negative class.
for (cell in getDetectionObjects()) {
    def finalClass = outputClasses[numClasses];
    def finalColor = outputColors[numClasses];
    def bestScore = 0;
    for (int i = 0; i < numClasses; i++) {
        if (maxValues[i] > 0 && PathClassTools.containsName(cell.getPathClass(), classes[i])) {
            def value = cell.getMeasurementList().getMeasurementValue(measurements[i]);
            def score = value / maxValues[i];  // Weighted score value
            if (score > bestScore) {
                finalClass = outputClasses[i];
                finalColor = outputColors[i];
                bestScore = score;
            }
        }
    }
    cell.setClassifications([finalClass]);
    cell.getPathClass().setColor(finalColor);
}
fireHierarchyUpdate();

// Calculate distance and neighborhood measurements
detectionCentroidDistances(true);
selectAnnotations();
runPlugin('qupath.opencv.features.DelaunayClusteringPlugin', '{"distanceThresholdMicrons":0.0,"limitByClass":false,"addClusterMeasurements":false}');

// Save region data (as GeoJSON) and point data (as CSV)
exportSelectedObjectsToGeoJson(outputDir + outputName + ".json", "PRETTY_JSON", "FEATURE_COLLECTION")
saveDetectionMeasurements(outputDir + outputName + ".csv");
