import React from "react";
import ReactDOM from "react-dom/";
import { HashRouter, Route, Switch, Redirect } from "react-router-dom";
import { ThemeProvider } from '@mui/material/styles';
import { CssBaseline } from '@mui/material';

// Import the new theme
import theme from './theme';

// styles for this kit
import "assets/css/bootstrap.min.css";
import "assets/scss/now-ui-kit.scss?v=1.5.0";
import "assets/demo/demo.css?v=1.5.0";
import "assets/demo/nucleo-icons-page-styles.css?v=1.5.0";

// pages for this kit
import Index from "views/Index.js";
import DocumentPage from "views/examples/DocumentPage";
// import LoginPage from "views/examples/LoginPage.js";
// import LandingPage from "views/examples/LandingPage.js";
import ProfilePage from "views/examples/ProfilePage.js";
import Demo from "views/test";
import Gallery from "views/examples/Gallery";
import Chatbot from "views/examples/Chatbot";
import TradePage from "views/examples/TradePage";

// const root = ReactDOM.createRoot(document.getElementById("root"));

ReactDOM.render(
  <ThemeProvider theme={theme}>
    <CssBaseline />
    <HashRouter>
      <Switch>
        <Switch>
          <Route path="/index" render={(props) => <Index {...props} />} />
          <Route
            path="/documents"
            render={(props) => <DocumentPage {...props} />}
          />
          <Route
            path="/profile-page"
            render={(props) => <ProfilePage {...props} />}
          />
          <Route
            path="/test"
            render={(props) => <Demo {...props} />}
          />
          <Route
            path="/gallery"
            render={(props) => <Gallery {...props} />}
          />
          <Route
            path="/trades"
            render={(props) => <TradePage {...props} />}
          />
          <Route
            path="/chat"
            render={(props) => <Chatbot />}
          />

          <Redirect to="/index" />
          <Redirect from="/" to="/index" />
        </Switch>
      </Switch>
    </HashRouter>
  </ThemeProvider>,
  document.getElementById('root')
);