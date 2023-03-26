// component
// import Iconify from '../../components/Iconify';
import { Icon } from '@iconify/react';

// ----------------------------------------------------------------------

// const getIcon = (name) => <Iconify icon={name} width={22} height={22} />;

const sidebarConfig = [
  // {
  //   title: 'dashboard',
  //   path: '/dashboard',
  //   icon: getIcon('eva:pie-chart-2-fill')
  // },
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
    title: 'contacts',
    path: '/contacts',
    icon: <Icon icon="ic:baseline-perm-contact-calendar" color="#420" width={22} height={22} />
  }
];

export default sidebarConfig;
