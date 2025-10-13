async function loadModel() {
    const MODEL_URL = './' + window.folder + '/model_classifier_CNN_js/model.json';
    try {
        const model = await tf.loadLayersModel(MODEL_URL);
        console.log("✅ Model loaded successfully from the fixed file!");
        model.summary();
        return model;
    } catch (error) {
        console.error("CRITICAL: Final load attempt failed:", error);
    }
}

/*
async function loadModel() {
    console.log("FOLDER3:", window.folder);
    console.log("Model input shape:", window.input_shape);
    
    // NOTE: I'm using your currently active path.
    const MODEL_URL = './' + window.folder + '/model_classifier_CNN_js/model.json';

    let model; // Declare 'model' here to store the result of loadLayersModel

    try {
        // 1. Load the model architecture and weights.
        // This is the line that throws the "InputLayer" error, 
        // but it still returns a model object with layers.
        model = await tf.loadLayersModel(MODEL_URL);
        
        // 2. Define a new, explicit Input layer with the correct shape.
        const fixedInput = tf.input({ shape: window.input_shape });
        
        // 3. Trace the output of the *successfully loaded* model (now named 'model')
        // by feeding it the new input.
        const fixedOutput = model.apply(fixedInput);
        
        // 4. Create a new model instance using the new Input and the old Output.
        const finalModel = tf.model({
            inputs: fixedInput,
            outputs: fixedOutput
        });
        
        console.log("✅ Model successfully loaded and re-wrapped with explicit input shape.");
        finalModel.summary();
        
        return finalModel;
        
    } catch (error) {
        // If the error persists even after re-wrapping, there might be a deeper 
        // issue in the model's layers (e.g., custom layers or shape dependency 
        // that re-wrapping can't fix).
        console.error("Error loading or re-wrapping the model:", error);
    }
}

*/

function showData() {
    let R = window.pyscriptData.toJs({ dict_converter: Object.fromEntries })[0];
    console.log("R:", R);
    console.log("R.length:",R.length);
    console.log("FOLDER:",window.folder);
    const pred_value = 123;
    const mineral_name = window.py_getMineral(folder+"/AAA_table_names.h5", pred_value);
    console.log("MINERAL:",mineral_name);
}

function selectModel() {
  setCookie("selectedIndex", document.SpectraKeras.model.selectedIndex ,1000);
 }

function init() {
  document.SpectraKeras.model.selectedIndex = getCookie("selectedIndex");
}

window.onload = init;

// #######  Utilities  ##################################
function getCookie(name) {
  return (name = (document.cookie + ';').match(new RegExp(name + '=.*;'))) && name[0].split(/=|;/)[1];
}

function setCookie(name, value, days) {
  var e = new Date;
  e.setDate(e.getDate() + (days || 365));
  document.cookie = name + '=' + value + ';expires=' + e.toUTCString() + ';path=/;domain=.' + document.domain;
}
// ########################################################
