async function runModel() {
    console.log("FOLDER3:",window.folder);
    //const MODEL_URL = './'+window.folder+'/model_classifier_CNN_js/model.json';
    const MODEL_URL = './'+window.folder+'/tfjs_output_dir/model.json';

    try {
        const model = await tf.loadLayersModel(MODEL_URL);
        console.log("Model is ready for prediction!");

        // Example usage:
        // const prediction = model.predict(yourInputTensor); 
        // prediction.print();
    } catch (error) {
        console.error("Error loading the model:", error);
    }
    
    let R = window.pyscriptData.toJs({ dict_converter: Object.fromEntries })[0];
    console.log("R:", R);
    console.log("R.length:",R.length);
    console.log("FOLDER:",window.folder);
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
