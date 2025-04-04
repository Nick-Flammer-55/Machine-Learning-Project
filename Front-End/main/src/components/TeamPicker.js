import React from 'react';
import { Select, MenuItem, FormControl, InputLabel } from '@mui/material';

function TeamPicker(props) {

  const imagePath = `/assets/team_logos/${props.team}.svg`;

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
      <img src={imagePath} alt="" width={150}/>
      <FormControl>
        <InputLabel>Team</InputLabel>
        <Select
          labelId="team-select-label"
          id="team-select"
          value={props.team}
          label={"Team"}
          onChange={handleSelect}
          sx={{width: "200px"}}
          MenuProps={{
            PaperProps: {
              style: {
                maxHeight: 200, /* Max height of the dropdown */
                overflowY: 'auto', /* Enable vertical scrolling */
              },
            },
          }}
        >
          {props.opTeam !== "atl" && <MenuItem value={"atl"}>Atlanta Hawks</MenuItem>}
          {props.opTeam !== "bos" && <MenuItem value={"bos"}>Boston Celtics</MenuItem>}
          {props.opTeam !== "bkn" && <MenuItem value={"bkn"}>Brooklyn Nets</MenuItem>}
          {props.opTeam !== "cha" && <MenuItem value={"cha"}>Charlotte Hornets</MenuItem>}
          {props.opTeam !== "chi" && <MenuItem value={"chi"}>Chicago Bulls</MenuItem>}
          {props.opTeam !== "cle" && <MenuItem value={"cle"}>Cleveland Cavaliers</MenuItem>}
          {props.opTeam !== "dal" && <MenuItem value={"dal"}>Dallas Mavericks</MenuItem>}
          {props.opTeam !== "den" && <MenuItem value={"den"}>Denver Nuggets</MenuItem>}
          {props.opTeam !== "det" && <MenuItem value={"det"}>Detroit Pistons</MenuItem>}
          {props.opTeam !== "gsw" && <MenuItem value={"gsw"}>Golden State Warriors</MenuItem>}
          {props.opTeam !== "hou" && <MenuItem value={"hou"}>Houston Rockets</MenuItem>}
          {props.opTeam !== "ind" && <MenuItem value={"ind"}>Indiana Pacers</MenuItem>}
          {props.opTeam !== "lac" && <MenuItem value={"lac"}>Los Angeles Clippers</MenuItem>}
          {props.opTeam !== "lal" && <MenuItem value={"lal"}>Los Angeles Lakers</MenuItem>}
          {props.opTeam !== "mem" && <MenuItem value={"mem"}>Memphis Grizzlies</MenuItem>}
          {props.opTeam !== "mia" && <MenuItem value={"mia"}>Miami Heat</MenuItem>}
          {props.opTeam !== "mil" && <MenuItem value={"mil"}>Milwaukee Bucks</MenuItem>}
          {props.opTeam !== "min" && <MenuItem value={"min"}>Minnesota Timberwolves</MenuItem>}
          {props.opTeam !== "nop" && <MenuItem value={"nop"}>New Orleans Pelicans</MenuItem>}
          {props.opTeam !== "nyk" && <MenuItem value={"nyk"}>New York Knicks</MenuItem>}
          {props.opTeam !== "okc" && <MenuItem value={"okc"}>Oklahoma City Thunder</MenuItem>}
          {props.opTeam !== "orl" && <MenuItem value={"orl"}>Orlando Magic</MenuItem>}
          {props.opTeam !== "phi" && <MenuItem value={"phi"}>Philadelphia 76ers</MenuItem>}
          {props.opTeam !== "phx" && <MenuItem value={"phx"}>Phoenix Suns</MenuItem>}
          {props.opTeam !== "por" && <MenuItem value={"por"}>Portland Trailblazers</MenuItem>}
          {props.opTeam !== "sac" && <MenuItem value={"sac"}>Sacramento Kings</MenuItem>}
          {props.opTeam !== "sas" && <MenuItem value={"sas"}>San Antonio Spurs</MenuItem>}
          {props.opTeam !== "tor" && <MenuItem value={"tor"}>Toronto Raptors</MenuItem>}
          {props.opTeam !== "uta" && <MenuItem value={"uta"}>Utah Jazz</MenuItem>}
          {props.opTeam !== "was" && <MenuItem value={"was"}>Washington Wizards</MenuItem>}
        </Select>
      </FormControl>
    </div>
  );
}

export default TeamPicker;