import React from 'react';
import { Select, MenuItem, FormControl, InputLabel } from '@mui/material';
import teamAssets from './team_info.json';

function TeamPicker(props) {

  const getTeam = (team) => {
    return (teamAssets.teams.find(item => item.team === team))
  };

  const handleSelect = (e) => {
    props.setTeam(e.target.value)
  };

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center'
      }}
    >
      {(props.team !== null) && <img src={props.team.logo} alt="" width={150}/>}
      <FormControl>
        <InputLabel>Team</InputLabel>
        <Select
          labelId="team-select-label"
          id="team-select"
          value={props.team}
          label={"Team"}
          onChange={handleSelect}
          sx={{width: "250px"}}
          MenuProps={{
            PaperProps: {
              style: {
                maxHeight: 200, /* Max height of the dropdown */
                overflowY: 'auto', /* Enable vertical scrolling */
              },
            },
          }}
        >
          {teamAssets.teams.map((nbateam) => 
            (props.opTeam !== getTeam(nbateam.team) && 
            <MenuItem key={nbateam.team} 
              value={getTeam(nbateam.team)}
            >
              {nbateam.full_name}
            </MenuItem>
          ))}
        </Select>
      </FormControl>
    </div>
  );
}

export default TeamPicker;