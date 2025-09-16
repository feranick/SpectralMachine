<?php

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    if (isset($_FILES['files'])) {
        $errors = [];
        $path = 'uploads/';
        $extensions = ['txt'];

        $files = $_FILES['files']['tmp_name'];
        $files_names = $_FILES['files']['name'];
        $num_files = count($_FILES['files']['tmp_name']);

        for ($i = 0; $i < $num_files; $i++) {
            $file_name = $_FILES['files']['name'][$i];
            $file_tmp = $files[$i];
            $file_type = $_FILES['files']['type'][$i];
            $file_size = $_FILES['files']['size'][$i];
            $file_ext = strtolower(end(explode('.', $_FILES['files']['name'][$i])));

            $file = $path . $file_name;

            if (!in_array($file_ext, $extensions)) {
                $errors[] = 'Extension not allowed: ' . $file_name . ' ' . $file_type;
            }

            if ($file_size > 2097152) {
                $errors[] = 'File size exceeds limit: ' . $file_name . ' ' . $file_type;
            }

            if (empty($errors)) {
                move_uploaded_file($file_tmp, $file);
            }
        }

        $tmpfile = $files[0];
        if ($_POST['mode'] == "Raman Spectroscopy") {
            $folder = "ml-raman";
            }
        if ($_POST['mode'] == "Powder X-ray Diffraction (XRD)") {
            $folder = "ml-xrd";
            }

        if ($num_files == 1) {
            //$command = "cd " . $folder . "; SpectraKeras_CNN -p $tmpfile 2>&1";
            $command = "cd " . $folder . "; export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-12.5; python3 ../SpectraKeras_CNN.py -p $tmpfile 2>&1";
            $output = shell_exec($command);
        }
        else {
            $f = json_encode($files);
            $fn = json_encode($files_names);
            $command = "cd " . $folder . "; export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-12.5; python3 ../SpectraKeras_CNN.py -b $f $fn 2>&1";
            $output = shell_exec($command);
        }

        if ($errors) print_r($errors);
    }
}

?>

<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <meta http-equiv="X-UA-Compatible" content="ie=edge" />

    <title>SpectraKeras</title>
    <script type="text/javascript" src="SpectraKeras.js" ></script>
  </head>

  <body>
    <script>
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


    </script>

    <h2>SpectraKeras: Identify  minerals from spectra</h2>

    Upload one or more ASCII files with the Raman spectra or an XRD scan of an uknown mineral. A Convolutional neural netowrk machine learning algorithm trained on the <a href="https://rruff.info" target="_blank" rel="noopener noreferrer">Rruff library</a> will predict the type of mineral.
    <br>Files are discarded after prediction. <a href="https://github.com/feranick/SpectralMachine" target="_blank" rel="noopener noreferrer">SpectraKeras is open-source and code and python scripts are available on Github</a>.
    <br><br> Sample input files can be found here for <a href="ml-raman/Abelsonite_raman.txt">raman</a> or <a href="ml-xrd/Albite_xrd.txt">powder-xrd</a>.
    <br><br>To get names of the minerals corresponding to the prediction values, use the link to the ML models below.
    <br>Current Raman ML model: <a href="ml-raman/AAA-20250413r_2025-04-14_11-42-21.csv">AAA-20250413_norm1_train-cv_hfsel20_val37 CNN_2-15_b4_keras3 </a>
    <br>Current XRD ML model: <a href="ml-xrd/AAA-powder-20250225_2025-02-25_16-07-01.csv">AAA-Powder_20250225_norm1_train-cv_hfsel10_val22 CNN_powder_2-15_b4</a>

    <form name="SpectraKeras" method="post" enctype="multipart/form-data">
      <br><br><input type="file" name="files[]" multiple />
      <input type="submit" value="Identify Mineral via ML" name="submit" />
      <br><br><select name="mode" id="mode" onchange="selectModel()">
            <option>Raman Spectroscopy</option>
            <option>Powder X-ray Diffraction (XRD)</option>
            </select>
    </form>
    <text_area><pre><?php echo $output; ?></pre></text_area>
    <text_area><pre><?php echo $output2; ?></pre></text_area>

  </body>
</html>
