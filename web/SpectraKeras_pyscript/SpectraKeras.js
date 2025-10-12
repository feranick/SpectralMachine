function showPyData() {
    // 1. Retrieve the Pyodide Proxy object
    let R = window.pyscriptData.toJs({ dict_converter: Object.fromEntries })[0];
    
    // 3. Access and display the data
    console.log("R:", R);
    console.log("R.length:",R.length);
    console.log(window.folder);
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
