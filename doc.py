import numpy as np
import pandas as pd
import calendar as cd
import datetime as dt
import pdb
 
 
class DocSchedulingProblem:
    """This class encapsulates the Nurse Scheduling problem
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

    def getDaysInMonth(self):
        return self.days_in_month



 
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

def getInitShedule(doc):
#1 Взять случайного дежуранта из списка ✓
#2 Рассчитать среднее количество дежурств на дежуранта
#3 Генерировать максимально разбросанные даты дежурств
#4 Проверить совпадения по уже поставленным с учетом корпусов, 	корпуса можно делить поровну
#5 Если есть совпадения, сдвинуть на один день
#6 Повторить пункт 4
#7 Проверить минимальное расстояние между дежурствами
 
#    pdb.set_trace()
    #количество дежурств в следующем месяце,
    #умноженное на количество дежурантов - 
    shifts = len(doc)


    #генерировать массив нулей
    schedule = np.zeros(shifts,dtype=np.int8)
    
    #генерировать случайное число от 0 до количества
    #дежурантов
    l = np.random.randint(0,len(doc.getRealDejs()))

    #получить случайного дежуранта с этим индексом
    d = doc.getRealDejs().iloc[l] 
    print(l,d)

    #сдвиг для получения профиля каждого дежуранта
    shift_schedule = doc.corps*doc.days_in_month


    schedule[l*shift_schedule]=1
    print(type(schedule[5:24]))

    print(l,d)


    return schedule


def getFreeDejFromSchedule(schedule,days_in_month,nmb_corps):
    #отдает в виде массива свободные дежурства

    nmb_dej = int(len(schedule)/(days_in_month*nmb_corps))
#    pdb.set_trace()

    ar = schedule.reshape(nmb_dej, days_in_month*nmb_corps)
    sum = ar.sum(axis=0)

    return np.where(sum ==0)




 
 
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
 
 
