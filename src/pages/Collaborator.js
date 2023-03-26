import { sentenceCase } from 'change-case';

import {
  Card,
  Table,
  Stack,
  Avatar,
  TableRow,
  TableHead,
  TableBody,
  TableCell,
  Container,
  Typography,
  TableContainer
} from '@mui/material';
// components
import Page from '../components/Page';
import Label from '../components/Label';
import Scrollbar from '../components/Scrollbar';
import { imgAvatar } from '../utils/avatars'

// ----------------------------------------------------------------------

function createData(id, name, email, status, role) {
  const avatarUrl = imgAvatar(name);
  return { id, name, email, status, role, avatarUrl };
}

const users = [
  createData(1, 'Changheng Zhou', 'changhengz@student.unimelb.edu.au', 'active', ''),
  createData(2, 'Sirui Wang', 'sirui@student.unimelb.edu.au', 'active', ''),
  createData(3, 'Mian Chen', 'mianc@student.unimelb.edu.au', 'active', ''),
  createData(4, 'Yuheng Fan', 'yuhengf@student.unimelb.edu.au', 'active', ''),
  createData(5, 'Zhanzhao Yang', 'yangzhanzhao1994@gmail.com', 'active',  ''),
]
// ----------------------------------------------------------------------

export default function Collaborator() {

  // const colorOf = (role) => {
  //   if (role === "Front End Developer") {
  //     return "warning";
  //   } else if (role === "Backend Developer") {
  //     return "info";
  //   } 
  // };

  return (
    <Page title="Collaborators | COMP90024-Group 21">
      <Container maxWidth='xl'>

        <Stack direction="row" alignItems="center" justifyContent="space-between" mb={5}>
          <Typography variant="h4" gutterBottom>
            Team Members
          </Typography>
          
        </Stack>

        <Card>

          <Scrollbar>
            <TableContainer sx={{ minWidth: 800 }}>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell align="left">Name</TableCell>
                    <TableCell align="left">email</TableCell>
                    {/* <TableCell align="left">role</TableCell> */}
                    <TableCell align="left">status</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {users.map((row) => (
                    <TableRow 
                      hover
                      key={row.id}
                    > 
                      <TableCell align="left">
                        <Stack direction="row" alignItems="center" spacing={2}>
                          <Avatar alt={row.name} src={row.avatarUrl} />
                          <Typography variant="subtitle2" noWrap >
                            {row.name}
                          </Typography>
                        </Stack>
                      </TableCell>
                      <TableCell align="left">{row.email}</TableCell>
                      <TableCell align="left">
                            <Label
                              variant="ghost"
                              color={(row.status === 'banned' && 'error') || 'success'}
                            >
                              {sentenceCase(row.status)}
                            </Label>
                          </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Scrollbar>
        </Card>
      </Container>
    </Page>
  );
}
