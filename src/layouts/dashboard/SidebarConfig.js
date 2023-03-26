// component
import Iconify from '../../components/Iconify';

// ----------------------------------------------------------------------

const getIcon = (name) => <Iconify icon={name} width={22} height={22} />;

const sidebarConfig = [
  {
    title: 'dashboard',
    path: '/COMP90024_Group21/dashboard',
    icon: getIcon('eva:pie-chart-2-fill')
  },
  // {
  //   title: 'map',
  //   path: '/COMP90024_Group21/map',
  //   icon: getIcon('eva:map-fill')
  // },
  // {
  //   title: 'team members',
  //   path: '/COMP90024_Group21/collaborator',
  //   icon: getIcon('eva:people-fill')
  // },
  {
    title: 'project link',
    path: '/COMP90024_Group21/projectlinks',
    icon: getIcon('eva:github-fill')
  }
];

export default sidebarConfig;
