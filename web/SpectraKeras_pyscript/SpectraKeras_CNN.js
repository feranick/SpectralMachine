var spectra = []

function handleFileSelect(){
  var file = document.getElementById("file").files[0];
  var reader = new FileReader();
  reader.onload = function(file) {
    lines = file.target.result.split('\n');
    const result = [];

    for (const line of lines) {
      const columns = line.split('\t').map(Number);
      if (columns.length >= 2) {
        spectra.push([columns[0], columns[1]]);
      }
    }
    console.log(spectra);
    };
  reader.readAsText(file);
 }
