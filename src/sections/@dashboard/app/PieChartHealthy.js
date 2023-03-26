import React, {useEffect, useState} from 'react';
import ReactECharts from 'echarts-for-react';
import { styled } from '@mui/material/styles';
import { Card, CardHeader, Select, MenuItem, InputLabel, FormControl } from '@mui/material';
import colors from 'src/utils/colorSeries';
import SERVER from './config'; 

const CHART_HEIGHT = 392;
const LEGEND_HEIGHT = 72;

const ChartWrapperStyle = styled('div')(({ theme }) => ({
  height: CHART_HEIGHT,
  marginTop: theme.spacing(2),
  '& .apexcharts-canvas svg': {
    height: CHART_HEIGHT
  },
  '& .apexcharts-canvas svg,.apexcharts-canvas foreignObject': {
    overflow: 'visible'
  },
  '& .apexcharts-legend': {
    height: LEGEND_HEIGHT,
    alignContent: 'center',
    position: 'relative !important',
    borderTop: `solid 1px ${theme.palette.divider}`,
    top: `calc(${CHART_HEIGHT - LEGEND_HEIGHT}px) !important`
  }
}));

export default function PieChartHealthy() {
  
  const [chartData, setChartData ] = useState([]);
  const [fields, setFields] = useState([]);
  const [indicator, setIndicator] = useState("");

  useEffect(() => {
    fetch(`${SERVER}/api/fields/healthy`, {
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
        setIndicator(data.data[0]);
      }
    )
  }, []);

  useEffect(() => {
    fetch(`${SERVER}/api/overview/pie/healthy?indicator=${indicator}`, {
      method: 'GET',
    }).then(
      response => response.json()
    ).then(
      data => {
        const CHART_DATA = [];    
        data.data.map((v) => {
          CHART_DATA.push({value: v[0], name: v[1]});
          return null;
        });
        
        setChartData(CHART_DATA);
      }
    )
  }, [indicator]);

  const options = {
    tooltip: {
      trigger: 'item'
    },
    legend: {
      orient: 'vertical',
      left: 20
    },
    series: [
      {
        // name: 'Access From',
        color: colors,
        type: 'pie',
        radius: '70%',
        center: ['55%', '60%'],
        data: chartData,
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowOffsetX: 0,
            shadowColor: 'rgba(0, 0, 0, 0.5)'
          }
        }, 
      }
    ]    
  };


  return (
    <Card>
      <CardHeader title="Healthy Condition (Top 10 Suburbs)" />
        <ChartWrapperStyle dir="ltr">
        <ReactECharts option={options}/>
        </ChartWrapperStyle>
        <FormControl
          sx={{ m:1, minWidth: 120, marginLeft: "40%"}}
        >
          <InputLabel id="aspect-select">indicator</InputLabel>
          <Select
              label="indicator"
              value={indicator}
              onChange={e=>{setIndicator(e.target.value)}}
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
