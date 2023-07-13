# QuPath scripts


## Pre-requisites

QuPath 0.4.0+.


## File structure

- `segment_and_classify.groovy` Runs cell detection and applies pre-trained object classifiers.
- `resolve_ambiguous.groovy` Resolves ambiguous (multi-classed) cells after classification, by assigning the class or cell type corresponding to the most expressed marker after a normalization step.
- `object_classifiers/` Pre-trained object classifiers provided as examples.


## Usage

To run `segment_and_classify.groovy` on a single image or in batch mode:

- Create an empty QuPath project for the image or images.
- Load the object classifiers (.json files) via `Classify->Object classifier->Load object classifiers` you have previously trained.
- Drag-and-drop the script onto the QuPath window, and then run it from the Script Editor via `Run->Run` (for a single image) or `Run->Run for project` (for batch mode).

After running the script for cell detection and classification, you can run the second script `resolve_ambiguous.groovy` to resolve ambiguous cell types.
