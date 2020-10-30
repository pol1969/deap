import numpy as np
import pandas as pd
import calendar as cd
import datetime as dt
import pdb
import sys 
 
class DocSchedulingProblem:
    """This class encapsulates the Nurse Schedulizong problem
    """
 
    def __init__(self, hardConstraintPenalty,df_docs,month,year):
        """
        :param hardConstraintPenalty: the penalty factor for a hard-constraint violation
        """
        self.hardConstraintPenalty = hardConstraintPenalty
 
        # list of doc:
        
        self.docs = list(df_docs['FAM']+' '+ df_docs['NAME'])
        self.df = df_docs
        self.df_dej = self.getRealDejs()
        self.docs_dej = list(self.df_dej['FAM']+' '+ self.df_dej['NAME'])

       # doc' respective shift preferences - morning, evening, night:
        self.shiftPreference = [[1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 1, 1], [1, 1, 1]]
 
        # min and max number of doc allowed for each shift - morning, evening, night:
        self.shiftMin = [2, 2, 1]
        self.shiftMax = [3, 4, 2]
 
        # max shifts per week allowed for each doc
        self.maxShiftsPerWeek = 5
 
        # number of weeks we create a schedule for:
        self.weeks = 1
 
        # useful values:
        self.days_in_month = cd.monthrange(year,month)[1]
 #       pdb.set_trace()
        self.corps = 3
        self.month = month
        self.year = year
        self.realDejs = self.getRealDejs()
        self.shiftPerDay = len(self.shiftMin)
        self.shiftsPerWeek = 7 * self.shiftPerDay
        self.shiftsPerMonth = self.corps*self.days_in_month

    def getDfDocs(self):
        return self.df

    def getDocs(self):
        return self.docs

    def getCorps(self):
        return self.corps

    def getRealDejs(self):
        d = self.df.query('CORPUS!=2 and CORPUS!=0')
        return d
    def getNmbRealDejs(self):
        d = self.getRealDejs()
 #       pdb.set_trace()
        return d.shape[0]

    def getDaysInMonth(self):
        return self.days_in_month
    def getMonth(self):
        return self.month

    def getYear(self):
        return self.year



 
    def __len__(self):
        """
        :return: the number of shifts in the schedule

        """
 #       pdb.set_trace()
  #      return len(self.doc) * self.shiftsPerWeek * self.weeks
        return self.corps * self.days_in_month*self.realDejs.shape[0]
 
 
    def getCost(self, schedule):
        """
        Calculates the total cost of the various violations in the given schedule
        ...
        :param schedule: a list of binary values describing the given schedule
        :return: the calculated cost
        """
 
        if len(schedule) != self.__len__():
            raise ValueError("size of schedule list should be equal to ", self.__len__())
 
        # convert entire schedule into a dictionary with a separate schedule for each doc:
        docShiftsDict = self.getDocShifts(schedule)
 
        # count the various violations:
        consecutiveShiftViolations = self.countConsecutiveShiftViolations(docShiftsDict)
        shiftsPerWeekViolations = self.countShiftsPerWeekViolations(docShiftsDict)[1]
        docPerShiftViolations = self.countDocsPerShiftViolations(docShiftsDict)[1]
        shiftPreferenceViolations = self.countShiftPreferenceViolations(docShiftsDict)
 
        # calculate the cost of the violations:
        hardContstraintViolations = consecutiveShiftViolations+docPerShiftViolations + shiftsPerWeekViolations

        softContstraintViolations = shiftPreferenceViolations
 
        return self.hardConstraintPenalty * hardContstraintViolations + softContstraintViolations
 
    def getDocShifts(self, schedule):
        """
        Converts the entire schedule into a dictionary with a separate schedule for each doc
        :param schedule: a list of binary values describing the given schedule
        :return: a dictionary with each doc as a key and the corresponding shifts as the value
        """
 #       shiftsPerDoc = self.__len__() // len(self.realDejs)
        shiftsPerDoc = self.corps*self.days_in_month

        docShiftsDict = {}
        shiftIndex = 0
        #import pdb; pdb.set_trace()
 
        for doc in self.docs_dej:
            docShiftsDict[doc] = schedule[shiftIndex:shiftIndex + shiftsPerDoc]
            shiftIndex += shiftsPerDoc
 #       import pdb; pdb.set_trace()
 
        return docShiftsDict
 
    def countConsecutiveShiftViolations(self, docShiftsDict):
        """
        Counts the consecutive shift violations in the schedule
        Считает количество последовательных смен и штрафует за это
        :param docShiftsDict: a dictionary with a separate schedule for each doc
        :return: count of violations found
        """
        violations = 0
        # iterate over the shifts of each doc:
        for docShifts in docShiftsDict.values():
            # look for two cosecutive '1's:
   #         import pdb; pdb.set_trace()
            for shift1, shift2 in zip(docShifts, docShifts[1:]):
                if shift1 == 1 and shift2 == 1:
                    violations += 1
        return violations
 
    def countShiftsPerWeekViolations(self, docShiftsDict):
        """
        Counts the max-shifts-per-week violations in the schedule
        Считает количество смен в неделю и штрафует при превышении
        :param docShiftsDict: a dictionary with a separate schedule for each doc
        :return: count of violations found
        """
        violations = 0
        weeklyShiftsList = []
        # iterate over the shifts of each doc:
        for docShifts in docShiftsDict.values():  # all shifts of a single doc
            # iterate over the shifts of each weeks:
            for i in range(0, self.weeks * self.shiftsPerWeek, self.shiftsPerWeek):
                # count all the '1's over the week:
   #             import pdb; pdb.set_trace()
                weeklyShifts = sum(docShifts[i:i + self.shiftsPerWeek])
                weeklyShiftsList.append(weeklyShifts)
                if weeklyShifts > self.maxShiftsPerWeek:
                    violations += weeklyShifts - self.maxShiftsPerWeek
 
        return weeklyShiftsList, violations
 
    def countDocsPerShiftViolations(self, docShiftsDict):
        """
        Counts the number-of-doc-per-shift violations in the schedule
        :param docShiftsDict: a dictionary with a separate schedule for each doc
        :return: count of violations found
        """
        # sum the shifts over all doc:
        #pdb.set_trace()
        totalPerShiftList = [sum(shift) for shift in zip(*docShiftsDict.values())]
 
        violations = 0
        # iterate over all shifts and count violations:
        for shiftIndex, numOfDocs in enumerate(totalPerShiftList):
            dailyShiftIndex = shiftIndex % self.shiftPerDay  # -> 0, 1, or 2 for the 3 shifts per day
            if (numOfDocs > self.shiftMax[dailyShiftIndex]):
                violations += numOfDocs - self.shiftMax[dailyShiftIndex]
            elif (numOfDocs < self.shiftMin[dailyShiftIndex]):
                violations += self.shiftMin[dailyShiftIndex] - numOfDocs
 
        return totalPerShiftList, violations
 
    def countShiftPreferenceViolations(self, docShiftsDict):
        """
        Counts the doc-preferences violations in the schedule
        :param docShiftsDict: a dictionary with a separate schedule for each doc
        :return: count of violations found
        """
        violations = 0
#        import pdb; pdb.set_trace()
        for docIndex, shiftPreference in enumerate(self.shiftPreference):
            # duplicate the shift-preference over the days of the period
            preference = shiftPreference * (self.shiftsPerWeek // self.shiftPerDay)
            # iterate over the shifts and compare to preferences:
            shifts = docShiftsDict[self.docs_dej[docIndex]]
            for pref, shift in zip(preference, shifts):
                if pref == 0 and shift == 1:
                    violations += 1
 
        return violations
 
    def printScheduleInfo(self, schedule):
        """
        Prints the schedule and violations details
        :param schedule: a list of binary values describing the given schedule
        """
#        pdb.set_trace()
        docShiftsDict = self.getDocShifts(schedule)
 
        print("Schedule for each doc:")
        for doc in docShiftsDict:  # all shifts of a single doc
            print(doc, ":", docShiftsDict[doc])
 
        print("consecutive shift violations = ", self.countConsecutiveShiftViolations(docShiftsDict))
        print()
 
        weeklyShiftsList, violations = self.countShiftsPerWeekViolations(docShiftsDict)
        print("weekly Shifts = ", weeklyShiftsList)
        print("Shifts Per Week Violations = ", violations)
        print()
 
        totalPerShiftList, violations = self.countDocsPerShiftViolations(docShiftsDict)
        print("Docs Per Shift = ", totalPerShiftList)
        print("Docs Per Shift Violations = ", violations)
        print()
 
        shiftPreferenceViolations = self.countShiftPreferenceViolations(docShiftsDict)
        print("Shift Preference Violations = ", shiftPreferenceViolations)
        print()

def getInitSchedule(doc):
    """
    1 Взять случайного дежуранта из списка ✓
    2 Рассчитать среднее количество дежурств на дежуранта
    3 Генерировать максимально разбросанные даты дежурств
    4 Проверить совпадения по уже поставленным с учетом корпусов, 	корпуса можно делить поровну
    5 Если есть совпадения, сдвинуть на один день
    6 Повторить пункт 4
    7 Проверить минимальное расстояние между дежурствами
        """
    schedule = np.zeros(len(doc),dtype=np.int8)
    corps = doc.getCorps()
    days = doc.getDaysInMonth()
    nmb_dejs = doc.getNmbRealDejs()
    dejs = doc.getRealDejs()
    
    cnt=0
    while not isScheduleFull(schedule,doc):
        day = np.random.randint(1,days+1)
        dej = np.random.randint(0,nmb_dejs)
        corp = np.random.randint(1,corps+1)
 #       print()
 #       print(cnt,dejs.iloc[dej]['FAM'],day,corp)
  #      print(getDejsForDoc(schedule,doc,dej))
       
        i = convDejDayCorpToFlatten(doc,dej,day,corp)

        cnt+=1

        if isSuitableDej(schedule, doc,i,4):
#            print("Присваиваем: ")
 #           print(cnt,dejs.iloc[dej]['FAM'],day,corp)

            assignToDej(schedule,doc,dej,day,corp)
#            print('Присвоили')
    #        print(getDejsForDoc(schedule,doc,dej))

            printScheduleHuman(schedule,doc)
            printScheduleHumanSum(schedule,doc)
 #           pdb.set_trace()



        if cnt >2000:
            break

 #       input()
    printScheduleHuman(schedule,doc)
    printScheduleHumanSum(schedule,doc)



def isScheduleFull(schedule,doc):
    shifts = np.sum(schedule)
    shifts_max = doc.getCorps()*doc.getDaysInMonth()
#    print('Shifts ',shifts)
    if shifts < shifts_max:
        return False
    else:
        return True
  




def getFreeDejFromSchedule(schedule,days_in_month,nmb_corps):
    """
    отдает в виде массива свободные дежурства
    :schedule расписание
    :days_in_month дней в месяце
    :nmb_corps количество корпусов
    :return массив индексов свободных дежурств
    """

    #определяем число дежурантов
    nmb_dej = int(len(schedule)/(days_in_month*nmb_corps))

    #делаем из расписания двумерную матрицу
    ar = schedule.reshape(nmb_dej, days_in_month*nmb_corps)

    #считаем сумму в столбцах
    sum = ar.sum(axis=0)

    #возвращаем индексы дат,сумма дежурств в которых 
    #равна нулю (свободные дежурства) с учетом корпусов
    return np.where(sum==0)

def isSuitableDej(schedule, doc, sched_day,max_nmb_dej):
    """
    определяет, можно ли поставить день day в
    расписание shedule
    :shedule - расписание , двоичный массив длиной
      дней_в_месяце*количество_корпусов*количество_дежурантов
    :doc - объект DocSchedulingProblem
    :dej_index индекс дежуранта в массиве дежурантов
    :day индекс дня в одномерном массиве shedule
    :return True , если подходит

    """
#    pdb.set_trace()
    days = doc.getDaysInMonth()
    
    dej,day,corpus = convFlattenToDejDayCorp(doc,sched_day,days)
    dejs = doc.getRealDejs()
#    pdb.set_trace()
#    doc_dej = dejs[dej_index]
#    print(doc_dej
    
#    pdb.set_trace()
    
    if not isFreeDay(schedule,doc,sched_day):
 #       print("isFreeDay false")
        return False
    


    if not isSuitableCorpus(doc,dej,sched_day):
#        print("Не походит корпус")
        return False

   
    if not isSuitableQuantity(schedule,doc,dej,
            max_nmb_dej):
#        print("Не походит количество")
        return False


    if not isSuitableSequence(schedule, doc, sched_day):
#        print("Не походит последовательность")
        return False
#    print("Все подходит!")

    return True


def isSuitableSequence(schedule, doc, sched_day):

    dejs = doc.getRealDejs()
    days = doc.getDaysInMonth()
    dej,day,corp =  convFlattenToDejDayCorp(doc,sched_day,days)
    len_sched = len(schedule)

#    pdb.set_trace()
#    days_busy = getDejsForDoc(schedule,doc,dej)
    days_busy = getAmbitOne(schedule,doc,dej,3)
#    print(days_busy)
    
    if day in days_busy:
  #      pdb.set_trace()
        return False
    if day > len_sched :
        return False    

    return True


def isSuitableQuantity(schedule, doc, dej_index, max_nmb_dej):
 #   pdb.set_trace()
    dejs = doc.getRealDejs()
    days = doc.getDaysInMonth() 
    corps = doc.getCorps()
    nmb_max = days*corps
    num_rows, num_cols  = dejs.shape
    schedule = schedule.reshape(num_rows,nmb_max)
    schedule_doc = schedule[dej_index]
    #   dej_doc = dejs.iloc[dej_index]
    sum = np.sum(schedule_doc)

    if sum < max_nmb_dej:
        return True

    return False

def isSuitableCorpus(doc, dej_index, day):
 #   pdb.set_trace()

    df = doc.getRealDejs()
    dej_corp  = df.iloc[dej_index]['CORPUS']
    days = doc.getDaysInMonth()
    nmb_corps = doc.getCorps()
    possible_corpus = convFlattenToDejDayCorp(doc,day,days)[2] 
#    pdb.set_trace()
    return isCorpRight(dej_corp,possible_corpus) 
def isFreeDay(schedule,doc,sched_day):

    days = doc.getDaysInMonth()
    corps = doc.getCorps()
    dejs = doc.getNmbRealDejs()
#    pdb.set_trace()
    schedule2d = schedule.reshape(dejs,corps*days)
    schedule2d_sum = np.sum(schedule2d,axis=0)
    free_days = np.where(schedule2d_sum==0)[0]
    free_days =np.fromiter((x+1 for x in free_days),int)
    d = sched_day%(corps*days)+1

   # pdb.set_trace()
    if d in free_days: 
  #      print('Is free day')
  #      pdb.set_trace()
        return True 
    else:       
   #     print('Not is free day')
        return False

def assignToDej(schedule, doc, dej_index, day,corpus, flag=1):
    """
    присваивает (flag=1) и удаляет (flag=0) дежурства 
    дежурантам
    :schedule ДНК
    :doc объект DocSchedulingProblem
    :dej_index индекс дежуранта в массиве и т д
    :return плоская ДНК с изменениями
    """
    df = doc.getRealDejs()
    nmb_corps = doc.getCorps()
    nmb_dej_doc = len(doc.getRealDejs() )
    days = doc.getDaysInMonth()

    if day > days:
        print("Неправильный номер дня ",day)
        return 0
    if dej_index > nmb_dej_doc:
        print("Неправильный индекс дежуранта ", dej_index)
        return 0

    if corpus not in (1,2,3):
        print("Неправильный номер корпуса ",corpus)
        return 0

    if flag not in (0,1):
        print("Flag может принимать значения 0 и 1", flag)
        return 0

    #    pdb.set_trace()

    dej_doc = df.iloc[dej_index]

    schedule = schedule.reshape(nmb_dej_doc, days*nmb_corps)

        

    schedule[dej_index][(corpus-1)*days + day - 1] = flag

    return schedule.flatten()

def printScheduleHuman(schedule, doc):
    """
    Выводит расписание в нужном конечном формате
    :schedule ДНК расписания, двоичный 1d массив
    :doc - объект DocSchedulingProblem
    :return требуемая таблица
    """

    corps = doc.getCorps()
    days = doc.getDaysInMonth()
    fam = doc.getRealDejs()['FAM']

    # формируем столбик из дежурантов,
    # попутно добавляем к фамилии И О
    i = doc.getRealDejs()['NAME']
    o = doc.getRealDejs()['SURNAME']
    dejs =np.array( fam+' '+ i.str[0] +'.' +o.str[0] +'.')
    dejs = dejs.reshape((-1,1))
#    pdb.set_trace()
    # преобразуем ДНК в двумерный массив
    schedule = schedule.reshape((len(dejs),
        int(len(schedule)/len(dejs))))

    # цепляем слева столбец с ФИО дежурантов
    schedule = np.hstack((dejs,schedule))

    # можно сделать максимальный вывод таблицы без сокращений
    np.set_printoptions(threshold=sys.maxsize)

    # ищем в расписании единицы и получаем массивы координат
    #pdb.set_trace()
    is_one = np.where(schedule==1) 

    # начинаем формировать конечную матрицу
    # dtype object позволяет работать со строками
    hum = np.empty((days,3),dtype='object')
    
    # формируем массив дат для конечной таблицы
    dates = np.arange(np.datetime64(dt.date(doc.getYear(),
        doc.getMonth(),1)),days)
    # преобразуем его в столбец
    dates = dates.reshape(-1,1)

    # цепляем столбец с датами слева, попутно преобразуя
    # его в строку
    schedule_with_date = np.hstack(((dates.astype('str')),
        hum))

    # nditer позволяет итерировать двумерный массив
    # переносим данные в конечную таблицу,
    # попутно заполняя текстовые поля пробеллами
    # справа до 12
#    pdb.set_trace()
    try:
        it0 = np.nditer(is_one,flags=["multi_index"])
    except ValueError:
        print("printScheduleHuman передан пустой массив")
        return 0

    for y in it0:
  #      print()
    
       # print(y,it0.multi_index)

        dd = getDateFrom1d(y[1]-1,days,corps)
 #       print('Дата',dd)

        cc = getNmbCorpusFrom1d(y[1],days,corps)
 #       print('Корпус',cc)

 #       print('Дежурант',dejs[y[0]])

 #       print('Cell',schedule_with_date[dd][cc])

        schedule_with_date[dd][cc]= dejs[y[0]]
   


    
 #   for i in range(0,90):
 #
  #     print(i, getDateFrom1d(i,30,3),getNmbCorpusFrom1d(i,30,3))

    df = pd.DataFrame(schedule_with_date,columns=['Дата','2_корпус','1_корпус','Хоспис'])   
#    for i, j in df.iterrows():
 #       print(i,j[0],j[1],j[2],j[3])
  #  print()
    print("Расписание дежурств УЗ 'МООД'")
    print("Месяц ",doc.getMonth())
    print(df.to_string(index=False))
    free_shifts = np.sum(pd.isnull(schedule_with_date))
    print('Свободных смен - ',free_shifts)

#    pdb.set_trace()
    return schedule_with_date,free_shifts


def printScheduleHumanSum(schedule, doc):
    """
    Выводит расписание в нужном конечном формате
    :schedule ДНК расписания, двоичный 1d массив
    :doc - объект DocSchedulingProblem
    :return требуемая таблица
    """

    corps = doc.getCorps()
    days = doc.getDaysInMonth()
    fam = doc.getRealDejs()['FAM']

    # формируем столбик из дежурантов,
    # попутно добавляем к фамилии И О
    i = doc.getRealDejs()['NAME']
    o = doc.getRealDejs()['SURNAME']
    dejs =np.array( fam+' '+ i.str[0] +'.' +o.str[0] +'.')
    dejs = dejs.reshape((-1,1))

    # преобразуем ДНК в двумерный массив
    schedule2d = schedule.reshape((len(dejs),
        int(len(schedule)/len(dejs))))
#    pdb.set_trace()
    
    schedule_2d_sum = np.sum(schedule2d,axis=1)
    schedule2d_days = np.where(schedule2d==1)
#    pdb.set_trace()
    schedule_2d_sum = schedule_2d_sum.reshape((-1,1))
    all_sum = np.sum(schedule_2d_sum)
 #   pdb.set_trace()

    # цепляем слева столбец с ФИО дежурантов
    schedule_2d_sum_with_dejs = np.hstack((dejs,
        schedule_2d_sum))

    # можно сделать максимальный вывод таблицы без сокращений
    np.set_printoptions(threshold=sys.maxsize)
#    print(schedule_2d_sum_with_dejs)
    free_shifts = days*corps-all_sum
    print('Свободных смен ',free_shifts)
    return free_shifts

#    pdb.set_trace()

def getDejsForDoc(schedule, doc, dej_index):
    dejs = doc.getRealDejs()    
    days = doc.getDaysInMonth()
    corps = doc.getCorps()
    nmb_max = days*corps
    num_rows, num_cols  = dejs.shape
    schedule = schedule.reshape(num_rows,nmb_max)
#    pdb.set_trace()
    dd = np.where(schedule[dej_index]==1)[0]
    dd = (i + 1  for i in dd)
    #генерация numpy array from generator expression
    dd = np.fromiter(dd,int)

    return dd 



def getNmbCorpusFrom1d(nmb,nmb_days_in_month,nmb_corps):
    """
    :return номер корпуса по числу из ДНК - schedule
    """
 #   pdb.set_trace()
    if nmb<0 or nmb >nmb_days_in_month*nmb_corps:
        print("Неправильный номер дня", nmb)
 #       pdb.set_trace()
        return 0 
    if nmb <= nmb_days_in_month:
        return 1

    if nmb > nmb_days_in_month*2:
        return 3

    else:
        return 2

def getDateFrom1d(nmb,nmb_days_in_month,nmb_corps):
    """
    :return номер дня  по числу из ДНК - schedule
    """
    if nmb<0 or nmb >nmb_days_in_month*nmb_corps:
        print("Неправильный номер дня", nmb)
        return 0 
    if nmb < nmb_days_in_month:
        return nmb
    else:
        return nmb % nmb_days_in_month

def isCorpRight(dej_corp,possible_corpus):
    if possible_corpus not in (1,2,3):
        print('Неправильный номер корпуса -',possible_corpus)
        return False
    if dej_corp==1 and possible_corpus==3:
        return True
    if dej_corp==10 and possible_corpus==2:
        return True
    if dej_corp==100 and possible_corpus==1:
        return True
    if dej_corp==11 and possible_corpus in (2,3):
        return True
    if dej_corp==101 and possible_corpus in (1,3):
        return True
    if dej_corp==110 and possible_corpus in (1,2):
        return True
    if dej_corp==111 and possible_corpus in (1,2,3):
        return True

    return False   

def getAmbitOne(schedule,doc,dej,nmb_neighb,dej_before=0):
    """
    получить двоичный массив единиц дежурств с окрестностями
    :schedule исходный двоичный массив для одного дежуранта
    :дней в месяце
    :nmb_neighb число соседей с одной стороны,
        определяет размеры окрестности
    :dej_before BOOL было ли дежурство в последний день
        предыдущего месяца
    :return требуемый массив
    """
    dejs = doc.getRealDejs()    
    days = doc.getDaysInMonth()
    corps = doc.getCorps()
    nmb_max = days*corps
    num_rows, num_cols  = dejs.shape
    schedule = schedule.reshape(num_rows,nmb_max)
    schedule_doc = schedule[dej]

 #   pdb.set_trace()    
    x = len(schedule_doc)

    #количество корпусов
    rows  = int(x/days)

    #выстраиваем корпуса в столбик
    schedule_doc = schedule_doc.reshape(rows, days)

    #суммируем единицы по столбцам и получаем дни дежурств
    sum_schedule = np.sum(schedule_doc,axis=0)
#    print('In gAO sum_schedule:',sum_schedule)
    
    #получаем индексы дней дежурств
    ind = np.where(sum_schedule==1)
    un = np.array([],dtype='int16')
 #   print('Индексы единиц: ',ind)
    
    #получаем укaзатели соседних дней, когда нельзя
    #ставить дежурства 
    for i in ind[0]:
        p = np.arange(i-nmb_neighb+1,i+nmb_neighb+2)
        un = np.append(un,p)
#    print('Индексы единиц с соседями: ',un)
   
   #убираем повторы и отсекаем края
    ret = (x for x in np.unique(un) if x>0 and x<=days)
    ret = np.fromiter(ret, int)
#    print('Индексы без повторов и краев: ',ret)
#    pdb.set_trace()
   
    if dej_before==1:
        ret = np.insert(ret,0,0)

 #   print('Индексы с первым днем: ',ret) 


#    print('In getAmbientOne',ret)

    return ret

def convFlattenToDejDayCorp(doc,sched_day,days):
    """
    определяет по индексу в schedule дежуранта,день и корпус
    """
    corps = doc.getCorps()
    dejs = doc.getNmbRealDejs()
    mod = sched_day//days
    div = sched_day%days

    if div==0:
        return (sched_day//(days*corps),1,mod%corps+1)
    else:
        return (sched_day//(days*corps),div+1,mod%corps+1)


def convDejDayCorpToFlatten(doc,dej,day,corp):
    days = doc.getDaysInMonth()
    corps = doc.getCorps()
    dejs = doc.getNmbRealDejs()
   
    return dej*days*corps +(corp-1)*days + day-1



# testing the class:
def main():
    # create a problem instance:
    p = pd.read_csv("lk_1.csv")

    doc  = DocSchedulingProblem(10,p,dt.datetime.now().month+1,dt.datetime.now().year)
#    pdb.set_trace()
 
    randomSolution = getInitShedule(doc)

    print("Random Solution = ")
    print(randomSolution)
    print()
    print("Len randomSolution = ", len(randomSolution))
 
    doc.printScheduleInfo(randomSolution)
 
    print("Total Cost = ", doc.getCost(randomSolution))

 
if __name__ == "__main__":
    main()
 
 
