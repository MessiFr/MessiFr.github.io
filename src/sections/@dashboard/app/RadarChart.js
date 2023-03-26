import React, {useEffect, useState} from 'react';
import ReactECharts from 'echarts-for-react';
import { styled } from '@mui/material/styles';
import { Card, CardHeader } from '@mui/material';

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

export default function RadarChart() {
  
  const [ chartData, setChartData ] = useState([]);

  useEffect(() => {
    const dataBody = JSON.stringify({type: 'radar'});
    fetch('http://127.0.0.1:8000/getData/', {
      method: 'POST',
      body: dataBody
    }).then(
      response => response.json()
    ).then(
      data => {
        setChartData(data.data);
      }
    )
  }, []);

  

  const options = {
    legend: {
      data: ['Allocated Budget', 'Actual Spending'],
      left: 20,
      orient: 'vertical',
    },
    radar: {
      // shape: 'circle',
      indicator: chartData[1]
    },
    series: [
      {
        name: 'Budget vs spending',
        type: 'radar',
        data: chartData[0],
      }
    ]
  };

  return (
    <Card>
      <CardHeader title="Pie Chart" />
        <ChartWrapperStyle dir="ltr">
        <ReactECharts option={options}/>
        </ChartWrapperStyle>
    </Card>
  )
}
