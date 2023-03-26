import React, {useEffect, useState} from 'react';
import ReactECharts from 'echarts-for-react';
import { styled } from '@mui/material/styles';
import { Card, CardHeader, Select, MenuItem, InputLabel, FormControl } from '@mui/material';

const CHART_HEIGHT = 400;
const LEGEND_HEIGHT = 72;

const ChartWrapperStyle = styled('div')(({ theme }) => ({
  height: CHART_HEIGHT,
  marginTop: theme.spacing(2),
  '& .apexcharts-canvas svg': {
    height: CHART_HEIGHT
  },
  '& .apexcharts-canvas svg,.apexcharts-canvas foreignObject': {
    overflow: 'visible',
  },
  '& .apexcharts-legend': {
    height: LEGEND_HEIGHT,
    alignContent: 'center',
    position: 'relative !important',
    borderTop: `solid 1px ${theme.palette.divider}`,
    top: `calc(${CHART_HEIGHT - LEGEND_HEIGHT}px) !important`
  }
}));

export default function PieChart() {
  
  const [chartData, setChartData ] = useState([]);
  const [fields, setFields] = useState([]);
  const [chart1, setChart1] = useState("");
  const [chart2, setChart2] = useState("");
  const [aspect, setAspect] = useState("traffic");

  useEffect(() => {
    fetch("http://127.0.0.1:8000/getFields/", {
      method: 'GET'
    }).then(
      response => response.json()
    ).then(
      data => {
        const FIELDS = [];
        data.data.map((v) => {
          FIELDS.push(v);
          return null;
        });
        setFields(FIELDS);
        setChart1(data.data[0]);
        setChart2(data.data[1]);
      }
    )
    setAspect("traffic");
  }, []);


  useEffect(() => {
    const dataBody = JSON.stringify({type: 'pie', suburb: [chart1, chart2]});
    fetch('http://127.0.0.1:8000/getData/', {
      method: 'POST',
      body: dataBody
    }).then(
      response => response.json()
    ).then(
      data => {
        const CHART_DATA = [];    
        data.data.map((v) => {
          console.log(v)
          CHART_DATA.push(v);
          return null;
        });
        setChartData(CHART_DATA);
      }
    );
  }, [chart1, chart2]);

  const options = {
    // title: {
    //   text: 'chart1',
    //   left: 'center'
    // },
    color: ['#80FFA5', '#00DDFF', '#37A2FF', '#FF0087', '#FFBF00'],
    tooltip: {
      trigger: 'item'
    },
    legend: {
      orient: 'vertical',
      left: 20,
    },
    dataset: {
      source: chartData,
    },
    series: [
      {
        type: 'pie',
        radius: '70%',
        center: ['25%', '45%'],
        // data: chartData,
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowOffsetX: 0,
            shadowColor: 'rgba(0, 0, 0, 0.5)'
          }
        },
        encode: {
          itemName: 'suburb',
          value: chart1,
        } 
      },

      {
        type: 'pie',
        radius: '70%',
        center: ['75%', '45%'],
        // data: chartData,
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowOffsetX: 0,
            shadowColor: 'rgba(0, 0, 0, 0.5)'
          }
        },
        encode: {
          itemName: 'suburb',
          value: chart2,
        } 
      } 
    ]
  };

  return (
    <Card>
      <CardHeader title="Pie Charts" />
        <ChartWrapperStyle dir="ltr">
          <ReactECharts option={options} style={{height: "100%"}}/>
        </ChartWrapperStyle>
        <FormControl
          sx={{ m:1, minWidth: 120, marginLeft: "12%" }}
        >
          <InputLabel id="aspect-select">aspect</InputLabel>
          <Select
            labelId='aspect-select'
            label = "aspect"
            value={aspect}
            onChange={e=>{setAspect(e.target.value)}}
            style={{width:110, height:40}}
          >
            <MenuItem value={"traffic"}>Traffic</MenuItem>
            <MenuItem value={"healthy"}>Healthy</MenuItem>
          </Select>  
        </FormControl>
        <FormControl
          sx={{ m:1, minWidth: 120}}
        >
          <InputLabel id="aspect-select">suburb</InputLabel>
          <Select
              label="suburb"
              value={chart1}
              onChange={e=>{setChart1(e.target.value)}}
              style={{ width:200, height:40, marginLeft:"1%", marginBottom: 20 }}
          >
            { fields.map(item => (
              <MenuItem value={item}>{item}</MenuItem>
            ))}
          </Select>
        </FormControl>
        <FormControl
          sx={{ m:1, minWidth: 120, marginLeft: "34%"}}
        >
          <InputLabel id="aspect-select">suburb</InputLabel>
          <Select
              label="suburb"
              value={chart2}
              onChange={e=>{setChart2(e.target.value)}}
              style={{ width:200, height:40, marginLeft:"1%", marginBottom: 20 }}
          >
            { fields.map(item => (
              <MenuItem value={item}>{item}</MenuItem>
            ))}
          </Select>
        </FormControl>
        
        
    </Card>
  )
}
