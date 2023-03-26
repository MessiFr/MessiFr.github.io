// material
import { Box, Grid, Container, Typography } from '@mui/material';

// components
import Page from '../components/Page';
import {
  TagCloud,
  PieChartTraffic,
  PieChartHealthy,
  // SunBurstChart,
  // RadarChart,
  BarChart,
  Route2Map,
  HistoricalTweets,
  LiveTweets,
} from '../sections/@dashboard/app';


// ----------------------------------------------------------------------

export default function DashboardApp() {
  

  return (
    <Page title="Dashboard | COMP90024-Group 21">
      <Container maxWidth="xl">
        <Box sx={{ pb: 5 }}>
          <Typography variant="h4"> COMP90024 Group21</Typography>
        </Box>
        <Grid container spacing={3}>
          
          {/* <Grid item xs={12} md={6} lg={8}>
              <AppTest />
          </Grid> */}

          <Grid item xs={12} md={6} lg={6}>
            <PieChartTraffic />
          </Grid>

          <Grid item xs={12} md={6} lg={6}>
            <PieChartHealthy />
          </Grid>

          <Grid item xs={12} md={6} lg={12}>
            <LiveTweets />
          </Grid>

          <Grid item xs={12} md={6} lg={12}>
            <HistoricalTweets />
          </Grid>

          <Grid item xs={12} md={6} lg={12}>
            <BarChart />
          </Grid>

          <Grid item xs={12} md={6} lg={7}>
            <TagCloud />
          </Grid>
          <Grid item xs={12} md={6} lg={4}>
            <Route2Map />
          </Grid>
        </Grid>
      </Container>
    </Page>
  );
}
