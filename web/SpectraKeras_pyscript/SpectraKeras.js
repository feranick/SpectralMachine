function showPyData() {
    // 1. Retrieve the Pyodide Proxy object
    let R = window.pyscriptData.toJs({ dict_converter: Object.fromEntries })[0];
    
    // 3. Access and display the data
    console.log("R:", R);
    console.log("R.length:",R.length);
}

function selectModel() {
  setCookie("selectedIndex", document.SpectraKeras.mode.selectedIndex ,1000);
 }

function init() {
  document.SpectraKeras.mode.selectedIndex = getCookie("selectedIndex");
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


function setForm() {
    const url = 'index.php'
    const form = document.querySelector('form')
    form.addEventListener('submit', (e) => {
    e.preventDefault()
    const files = document.querySelector('[type=file]').files
    const formData = new FormData()
    for (let i = 0; i < files.length; i++) {
        let file = files[i]
        formData.append('files[]', file)
    }
    fetch(url, {
        method: 'POST',
        body: formData
    }).then(response => {
        return response.text();
    }).then(data => {
        console.log(data);
    });
    });
    }
