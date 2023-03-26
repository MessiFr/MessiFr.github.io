import { useRoutes, Navigate } from 'react-router-dom';
// layouts
import DashboardLayout from './layouts/dashboard';

//
import DashboardApp from './pages/DashboardApp'
import Collaborator from './pages/Collaborator'
import Map from './pages/Map'
import ProjectLink from './pages/Projectlink';
import Scenario1 from './pages/scenarios/scenario1';
import Scenario2 from './pages/scenarios/scenario2'
import NotFound from './pages/Page404';

// ----------------------------------------------------------------------

export default function Router() {
  return useRoutes([

    {
      path: '/',
      element: <DashboardLayout />,
      children: [
        { path: 'dashboard', element: <DashboardApp /> },
        { path: 'collaborator', element: <Collaborator /> },
        { path: 'map', element: <Map /> },
        { path: 'contacts', element: <ProjectLink /> }
      ]
    },
    {
      path: 'scenario1',
      element: <Scenario1 />,
    },
    {
      path: 'scenario2',
      element: <Scenario2 />,
    },
    {
      path: '/',
      element: <DashboardLayout />,
      children: [
        { path: '/', element: <Navigate to="/contacts" /> },
        { path: '404', element: <NotFound /> },
      ]
    },

  ]);
}
