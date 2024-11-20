var l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2"),
    l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"),
    srtm = ee.Image("USGS/SRTMGL1_003"),
    geometry = /* color: #d63000 */ee.Geometry.Point([91.02198552052914, 22.5851851402174]),
    water = /* color: #d63000 */ee.FeatureCollection(
        [ee.Feature(
            ee.Geometry.Polygon(
                [[[91.09731962300677, 22.56919938443167],
                  [91.09854271031756, 22.570843967508903],
                  [91.09772731877703, 22.571081736932772],
                  [91.09723379231829, 22.569694742845115],
                  [91.09731962300677, 22.56904086936343]]]),
            {
              "classvalue": 3,
              "system:index": "0"
            }),
        ee.Feature(
            ee.Geometry.Polygon(
                [[[91.20088545265126, 22.53058068865056],
                  [91.19779554786611, 22.539459726692442],
                  [91.18663755836415, 22.541520850293345],
                  [91.18663755836415, 22.53295905839302]]]),
            {
              "classvalue": 3,
              "system:index": "1"
            }),
        ee.Feature(
            ee.Geometry.Polygon(
                [[[91.13461342568463, 22.572371469007145],
                  [91.13503185029096, 22.57235165508955],
                  [91.13528934235639, 22.57260923579606],
                  [91.1344095777995, 22.572569608026384]]]),
            {
              "classvalue": 3,
              "system:index": "2"
            })]),
    built = /* color: #98ff00 */ee.FeatureCollection(
        [ee.Feature(
            ee.Geometry.Polygon(
                [[[91.12814010730004, 22.519397085823524],
                  [91.12841101041055, 22.51939213044269],
                  [91.12837077727532, 22.51970184140369],
                  [91.12815351834512, 22.51968449760822]]]),
            {
              "classvalue": 1,
              "system:index": "0"
            }),
        ee.Feature(
            ee.Geometry.Polygon(
                [[[91.08955181477454, 22.523106480266204],
                  [91.0895088994303, 22.523502899515666],
                  [91.0890582883158, 22.523344331952373],
                  [91.08899391529944, 22.52271005987934]]]),
            {
              "classvalue": 1,
              "system:index": "1"
            })]),
    dense_forest = 
    /* color: #0b4a8b */
    /* displayProperties: [
      {
        "type": "rectangle"
      },
      {
        "type": "polygon"
      },
      {
        "type": "polygon"
      }
    ] */
    ee.FeatureCollection(
        [ee.Feature(
            ee.Geometry.Polygon(
                [[[91.12718205762248, 22.52379435476187],
                  [91.12718205762248, 22.523521817224964],
                  [91.12721424413066, 22.523521817224964],
                  [91.12721424413066, 22.52379435476187]]], null, false),
            {
              "classvalue": 5,
              "system:index": "0"
            }),
        ee.Feature(
            ee.Geometry.Polygon(
                [[[91.12757366013865, 22.523492085824778],
                  [91.1274019987617, 22.523814175652674],
                  [91.12710159135203, 22.52379435476187],
                  [91.1272410662208, 22.523427667769095]]]),
            {
              "classvalue": 5,
              "system:index": "1"
            }),
        ee.Feature(
            ee.Geometry.Polygon(
                [[[91.11094297917158, 22.546678737534197],
                  [91.11094834358961, 22.546894253848816],
                  [91.11076058895857, 22.546881867862833],
                  [91.11077936442167, 22.546678737534197]]]),
            {
              "classvalue": 5,
              "system:index": "2"
            })]),
    bareland = /* color: #ffc82d */ee.FeatureCollection(
        [ee.Feature(
            ee.Geometry.Polygon(
                [[[91.1718850116904, 22.485521542980567],
                  [91.17581176568821, 22.489229026101217],
                  [91.17145585824802, 22.48710764417096]]]),
            {
              "classvalue": 2,
              "system:index": "0"
            })]),
    cropland = /* color: #00ffff */ee.FeatureCollection(
        [ee.Feature(
            ee.Geometry.Polygon(
                [[[91.1353223565859, 22.582120894824016],
                  [91.13512923753683, 22.583606825386855],
                  [91.13394906557028, 22.583547388472084],
                  [91.1335199121279, 22.58237845726987],
                  [91.13388469255392, 22.58073401183845]]]),
            {
              "classvalue": 4,
              "system:index": "0"
            }),
        ee.Feature(
            ee.Geometry.Polygon(
                [[[91.13510777986471, 22.58780696908908],
                  [91.13412072694723, 22.588203202449584],
                  [91.13343408143942, 22.58737111107589],
                  [91.13339116609518, 22.58673713332172],
                  [91.1347644571108, 22.586697509615185]]]),
            {
              "classvalue": 4,
              "system:index": "1"
            })]),
    roi = ee.FeatureCollection("projects/ee-piashchowdhury038/assets/central_coast_diss"),
    l7 = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2");



// Cloud mask function
function cloudMask(image){
    var qa = image.select('QA_PIXEL');
    var dilated = 1 << 1;
    var cirrus = 1 << 2;
    var cloud = 1 << 3;
    var shadow = 1 << 4;
    var mask = qa.bitwiseAnd(dilated).eq(0)
      .and(qa.bitwiseAnd(cirrus).eq(0))
      .and(qa.bitwiseAnd(cloud).eq(0))
      .and(qa.bitwiseAnd(shadow).eq(0));
    return image.select(['SR_B.*'], ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7'])
      .updateMask(mask)
      .multiply(0.0000275)
      .add(-0.2);
  }
  
  // Create an image composite
  var image = l8.filterBounds(roi).filterDate('2023-01-01', '2023-05-31')
    .merge(l9.filterBounds(roi).filterDate('2023-01-01', '2023-05-31'))
    .map(cloudMask)
    .median()
    .clip(roi);
  
  // Visualize
  Map.addLayer(image, { min: [0.1, 0.05, 0.05], max: [0.4, 0.3, 0.2], bands: ['B4', 'B65', 'B7']}, 'Image');
  
  // Band map
  var bandMap = {
    BLUE: image.select('B1'),
    GREEN: image.select('B2'),
    RED: image.select('B3'),
    NIR: image.select('B4'),
    SWIR1: image.select('B5'),
    SWIR2: image.select('B7')
  };
  
  // Add spectral indices
  var indices = ee.Image([
    { name: 'EVI', formula: '(2.5 * (NIR - RED)) / (NIR + 6 * RED - 7.5 * BLUE + 1)' },
    { name: 'NBR', formula: '(NIR - SWIR2) / (NIR + SWIR2)' },
    { name: 'NDMI', formula: '(NIR - SWIR1) / (NIR + SWIR1)' },
    { name: 'NDWI', formula: '(GREEN - NIR) / (GREEN + NIR)' },
    { name: 'NDBI', formula: '(SWIR1 - NIR) / (SWIR1 + NIR)' },
    { name: 'NDBaI', formula: '(SWIR1 - SWIR2) / (SWIR1 + SWIR2)' },
  ].map(function(dict){
    var indexImage = image.expression(dict.formula, bandMap).rename(dict.name);
    return indexImage;
  }));
  
  // Add index & SRTM to image
  image = image.addBands(indices).addBands(srtm.clip(roi));
  
  // Variable info
  var classValue = [1, 2, 3, 4, 5];
  var classNames = ['Built-up', 'Bareland', 'Water', 'Cropland', 'Dense_forest'];
  var classPalette = ['ff1344', '838b75', '1c07ff', 'f8ff3c', '00c213'];
  var columns = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'EVI', 'NBR', 'NDMI', 'NDWI', 'NDBI', 'NDBaI', 'elevation', 'classvalue', 'sample'];
  var features = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'EVI', 'NBR', 'NDMI', 'NDWI', 'NDBI', 'NDBaI', 'elevation'];
  
  // Sampels
  var samples = built.merge(bareland).merge(water).merge(cropland).merge(dense_forest)
    .map(function(feat){ return feat.buffer(30) });
  
  // Split samples to train and test per class
  samples = ee.FeatureCollection(classValue.map(function(value){
    var features = samples.filter(ee.Filter.eq('classvalue', value)).randomColumn();
    var train = features.filter(ee.Filter.lte('random', 0.8)).map(function(feat){ return feat.set('sample', 'train')});
    var test = features.filter(ee.Filter.gt('random', 0.8)).map(function(feat){ return feat.set('sample', 'test')});
    return train.merge(test);
  })).flatten();
  
  // Extract samples
  var extract = image.sampleRegions({
    collection: samples,
    scale: 30,
    properties: ['sample', 'classvalue']
  });
  
  // Train samples
  var train = extract.filter(ee.Filter.eq('sample', 'train'));
  print('Train sample size', train.size());
  var test = extract.filter(ee.Filter.eq('sample', 'test'));
  print('Test sample size', test.size());
  
  // Export image and samples
  Export.image.toDrive({
    image: image.toFloat(),
    scale: 30,
    maxPixels: 1e13,
    region: roi,
    crs: 'EPSG:4326',
    folder: 'DL',
    description: 'Landsat_Jambi_2023'
  });
  
  Export.table.toDrive({
    collection: extract,
    fileFormat: 'CSV',
    selectors: columns,
    description: 'Samples_LC_Jambi_2023',
    folder: 'DL'
  });
  
  // Random forest model
  var model = ee.Classifier.smileRandomForest(300).train(train, 'classvalue', features);
  print(model.explain());
  
  // Test model
  var cm = test.classify(model, 'predicted').errorMatrix('classvalue', 'predicted');
  print('Confusion matrix', cm, 'Accuracy', cm.accuracy(), 'Kappa', cm.kappa());
  
  // Apply model
  var lc = image.classify(model, 'lulc').clip(roi)
    .set('lulc_class_values', classValue, 'lulc_class_palette', classPalette);
  Map.addLayer(lc, {}, 'LULC');
  
  Export.image.toDrive({
    image: lc,
    description: 'Classified_Image_Jambi_2023',
    scale: 30,
    maxPixels: 1e13,
    region: roi,
    crs: 'EPSG:4326',
    folder: 'DL'});