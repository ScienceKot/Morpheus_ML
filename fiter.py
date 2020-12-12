from GAPlanner import GANPlanner

def filter_date(list_of_events):
    schedule = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    for event in list_of_events:
        schedule[int(event['start'].split('T')[1].split(':')[0])] = 0
        for i in range(int(event['start'].split('T')[1].split(':')[0]), int(event['end'].split('T')[1].split(':')[0])+1):
            try:
                schedule[i] = 0
            except IndexError:
                break
    return schedule
list_of_events = [{'start': '2020-12-13T08:30:00+02:00', 'end': '2020-12-13T09:30:00+02:00', 'summary': 'po sosati lu vdovicenco'}, {'start': '2020-12-13T22:30:00+02:00', 'end': '2020-12-13T23:30:00+02:00', 'summary': 'vecerniii blowjob de la vdovicenco'}]
filtered_data = filter_date(list_of_events)

gap = GANPlanner('forest.obj', 'DSI.obj')
best_schedule = gap.generate()