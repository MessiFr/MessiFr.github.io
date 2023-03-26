import { Container, Grid, Typography, Box } from "@mui/material";
import Page from "src/components/Page";
import { AppTrafficBySite } from "src/sections/@dashboard/app";

export default function ProjectLink() {
    return (
      <Page title="Contacts">
        
        <Container maxWidth='xl'>
          <Box sx={{ pb: 5 }}>
            <Typography variant="h4"> Contacts </Typography>
          </Box>
          <Grid item xs={12} md={6} lg={12}>
              <AppTrafficBySite />
            </Grid>
          </Container>
      </Page>
    );
  }
  