
var mapConfig2 = {
  "version": "v1",
  "config": {
    "visState": {
      "filters": [],
      "layers": [
        {
          "id": "0mnmnk2",
          "type": "geojson",
          "config": {
            "dataId": "melbourne_tweet_geo",
            "label": "Tweet Summary",
            "color": [
              34,
              63,
              154
            ],
            "highlightColor": [
              252,
              242,
              26,
              255
            ],
            "columns": {
              "geojson": "geometry"
            },
            "isVisible": true,
            "visConfig": {
              "opacity": 0.8,
              "strokeOpacity": 0.8,
              "thickness": 0.5,
              "strokeColor": [
                32,
                17,
                99
              ],
              "colorRange": {
                "name": "ColorBrewer RdPu-6",
                "type": "sequential",
                "category": "ColorBrewer",
                "colors": [
                  "#feebe2",
                  "#fcc5c0",
                  "#fa9fb5",
                  "#f768a1",
                  "#c51b8a",
                  "#7a0177"
                ]
              },
              "strokeColorRange": {
                "name": "Global Warming",
                "type": "sequential",
                "category": "Uber",
                "colors": [
                  "#5A1846",
                  "#900C3F",
                  "#C70039",
                  "#E3611C",
                  "#F1920E",
                  "#FFC300"
                ]
              },
              "radius": 10,
              "sizeRange": [
                0,
                10
              ],
              "radiusRange": [
                0,
                50
              ],
              "heightRange": [
                0,
                500
              ],
              "elevationScale": 5,
              "enableElevationZoomFactor": true,
              "stroked": true,
              "filled": true,
              "enable3d": false,
              "wireframe": false
            },
            "hidden": false,
            "textLabel": [
              {
                "field": null,
                "color": [
                  255,
                  255,
                  255
                ],
                "size": 18,
                "offset": [
                  0,
                  0
                ],
                "anchor": "start",
                "alignment": "center"
              }
            ]
          },
          "visualChannels": {
            "colorField": {
              "name": "sentiment",
              "type": "real"
            },
            "colorScale": "quantile",
            "strokeColorField": null,
            "strokeColorScale": "quantile",
            "sizeField": null,
            "sizeScale": "linear",
            "heightField": null,
            "heightScale": "linear",
            "radiusField": null,
            "radiusScale": "linear"
          }
        }
      ],
      "interactionConfig": {
        "tooltip": {
          "fieldsToShow": {
            "3w2mvmxgd": [
              {
                "name": "0",
                "format": null
              },
              {
                "name": "lga_name11",
                "format": null
              },
              {
                "name": "lga_code11",
                "format": null
              },
              {
                "name": "sentiment",
                "format": null
              },
              {
                "name": "smoking",
                "format": null
              }
            ]
          },
          "compareMode": false,
          "compareType": "absolute",
          "enabled": true
        },
        "brush": {
          "size": 0.5,
          "enabled": false
        },
        "geocoder": {
          "enabled": false
        },
        "coordinate": {
          "enabled": false
        }
      },
      "layerBlending": "normal",
      "splitMaps": [],
      "animationConfig": {
        "currentTime": null,
        "speed": 1
      }
    },
    "mapState": {
      "bearing": 0,
      "dragRotate": false,
      "latitude": -37.769791879463696,
      "longitude": 145.11904273646763,
      "pitch": 0,
      "zoom": 7.351805324662078,
      "isSplit": false
    },
    "mapStyle": {
      "styleType": "light",
      "topLayerGroups": {},
      "visibleLayerGroups": {
        "border": false,
        "building": true,
        "label": true,
        "land": true,
        "road": true,
        "water": true
      },
      "threeDBuildingColor": [
        9.665468314072013,
        17.18305478057247,
        31.1442867897876
      ],
      "mapStyles": {}
    }
  }
}

export default mapConfig2;