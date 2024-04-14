function bindGrid(response,selectedSpeciality,selectedProcedure) {
  let dates = response.dates;
  let values = response.forecast_values;
  $("#tbody").html("");
  for (var i = dates.length-1; i >=0 ; i--) {
    $('#tbody').append(`
          <tr>
            <td>${selectedSpeciality}</td>
            <td>${selectedProcedure}</td>
            <td>${dates[i]}</td>
            <td>${values[i]}</td>
          </tr>
    
    `)
  }
}
