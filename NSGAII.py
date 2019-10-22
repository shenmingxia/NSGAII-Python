import numpy as np

class NSGAII:
    def __init__(self, it=100, popSize=100, rc=0.8, rm=0.05, silent=True):
        self.silent = silent
        self.it = it
        self.popSize = popSize
        self.nc = round(rc * popSize * 2) / 2
        self.nm = round(rm * popSize)
        self.bestSolution = None
        self.bestObj = None
        self.NPS_solutions = []
        self.NPS_objs = []

    def Dominates(self, objList1, objList2):
        objList1 = np.array(objList1)
        objList2 = np.array(objList2)
        return all(objList1 <= objList2) and any(objList1 < objList2)

    def NonDominatedSorting(self, objsList):
        dominationList = []
        for i in range(len(objsList)):
            DominationSet = []
            DominatedCount = 0
            Rank = self.popSize
            dominationList.append([DominationSet, DominatedCount, Rank])
        F = []
        F.append([])  # F[0]

        for i in range(len(objsList) - 1):
            for j in range(i + 1, len(objsList)):
                if self.Dominates(objsList[i], objsList[j]):
                    dominationList[i][0].append(j)
                    dominationList[j][1] += 1
                elif self.Dominates(objsList[j], objsList[i]):
                    dominationList[j][0].append(i)
                    dominationList[i][1] += 1
            if dominationList[i][1] == 0:
                F[0].append(i)
                dominationList[i][2] = 1  # Rank

        k = 1
        while True:
            Q = []
            for i in F[k - 1]:
                for j in dominationList[i][0]:
                    dominationList[j][1] -= 1

                    if dominationList[j][1] == 0:
                        Q.append(j)
                        dominationList[j][2] = k + 1
            if not Q:
                break

            F.append(Q)
            k += 1
        return F, dominationList

    def CalcCrowdingDistance(self, objsList, F):
        CrowdingDistances = [0 for i in range(len(objsList))]

        nF = len(F)
        nObj = len(objsList[0])
        for k in range(nF):
            costs = []
            for kk in F[k]:
                costs.append(objsList[kk])
            n = len(F[k])
            d = np.zeros([n, nObj])
            objs = []
            for j in range(nObj):
                objs.append([])
                for pp in range(len(costs)):
                    objs[j].append(costs[pp][j])
                cj, so = self.sortedList(objs[j])

                d[so[0], j] = np.inf
                for i in range(1, n - 1):
                    if abs(cj[1] - cj[-1]) > 0:
                        d[so[i], j] = abs(cj[i + 1] - cj[i - 1]) / abs(cj[1] - cj[-1])
                    else:
                        d[so[i], j] = np.inf
                d[so[-1], j] = np.inf

            for i in range(len(objsList)):
                for j in range(len(F[k])):
                    if i == F[k][j]:
                        CrowdingDistances[i] = np.sum(d[j, :], axis=0)
        return CrowdingDistances

    def SortPopulation(self, pop, objs, crowdingDistance, dominationList):
        rank = []
        for i in range(len(dominationList)):
            rank.append(dominationList[i][2])
        # sort based on Crowding Distance
        rank2, _ = self.sortedFirstBySecond(rank, crowdingDistance, reverse=True)
        objs, _ = self.sortedFirstBySecond(objs, crowdingDistance, reverse=True)
        pop, _ = self.sortedFirstBySecond(pop, crowdingDistance, reverse=True)

        # sort based on Rank
        rank2, _ = self.sortedFirstBySecond(rank2, rank, reverse=False)
        objs, _ = self.sortedFirstBySecond(objs, rank, reverse=False)
        pop, _ = self.sortedFirstBySecond(pop, rank, reverse=False)

        # print('SortPopulation 1')
        # Update F
        maxRank = max(rank2)
        F = []
        for r in range(1, maxRank + 1):
            f = []
            for j in range(len(rank2)):
                if rank2[j] == r:
                    f.append(j)
            F.append(f)
        # print('SortPopulation 2')
        return pop, objs, F

    def optimize(self):
        self.pop = self.initial_solution()
        self.objs = self.objective_functions_minimization(self.pop)
        F, dominationList = self.NonDominatedSorting(self.objs)
        CrowdingDistances = self.CalcCrowdingDistance(self.objs, F)
        self.pop, self.objs, F = self.SortPopulation(self.pop, self.objs, CrowdingDistances, dominationList)

        NPS = []
        for p in range(0, self.it):
            # cross-over -------
            pop = self.pop
            for cr in range(1, round(self.nc / 2)):
                selection = self.tournament(2)
                parent1 = self.pop[selection[0]]
                parent2 = self.pop[selection[1]]
                child1, child2 = self.crossover(parent1, parent2)
                pop.append(child1)
                pop.append(child2)

            # # mutation-----
            for cr in range(1, round(self.nm)):
                selection = self.tournament(1)
                parent = self.pop[selection[0]]
                child = self.mutation(parent)
                pop.append(child)

            objs = self.objective_functions_minimization(pop)
            F, dominationList = self.NonDominatedSorting(objs)
            CrowdingDistances = self.CalcCrowdingDistance(objs, F)
            pop, objs, F = self.SortPopulation(pop, objs, CrowdingDistances, dominationList)

            # Truncate population
            self.pop = pop[:self.popSize]
            self.objs = objs[:self.popSize]
            F, dominationList = self.NonDominatedSorting(self.objs)
            CrowdingDistances = self.CalcCrowdingDistance(self.objs, F)
            self.pop, self.objs, F = self.SortPopulation(self.pop, self.objs, CrowdingDistances, dominationList)

            # store best objs
            objs = []
            average_objs = []
            NPS.append(0)
            nObj = len(self.objs[0])
            self.NPS_solutions = []
            self.NPS_objs = []
            for k in range(nObj):
                objs.append(0)
                average_objs.append(0)
                for i in range(self.popSize):
                    if dominationList[i][2] == 1:  # rank = 1
                        NPS[p] += 1
                        objs[k] += self.objs[i][k]
                        self.NPS_solutions.append(self.pop[i])
                        self.NPS_objs.append(self.objs[i])
                average_objs[k] = objs[k] / NPS[p]
            z = self.objs[F[0][0]]

            if not self.silent:
                print("it", str(p + 1), "objs", z, "NPS", str(NPS[p]), 'average_objs', average_objs)

        self.bestSolution = self.pop[0]
        self.bestObj = self.objs[0]

    def tournament(self, n):
        pdf = sorted([i for i in range(1, self.popSize + 1)], reverse=True)
        cdf = np.cumsum(pdf)
        selection = []
        for j in range(0, n):
            np.random.seed(None)
            r = np.random.random(1)
            for i in range(0, len(cdf)):
                if r <= cdf[i]:
                    selection.append(i)
                    break
        return selection

    def initial_solution(self):
        pass

    def objective_functions_minimization(self, pop):
        # just return a list of numbers for each obj value for minimization all of them
        pass

    def crossover(self, parent1, parent2):
        pass

    def mutation(self, parent):
        pass

    def sortedFirstBySecond(self, first, second, reverse=False):
        index = np.array(sorted(range(len(second)), key=lambda k: second[k], reverse=reverse))
        second = np.array(sorted(second, reverse=reverse))
        first = np.array(first)
        first = first[index]
        first = first.tolist()
        second = second.tolist()
        return first, second

    def sortedList(self, first, reverse=False):
	index = [i for i in range(len(first))]
        index, first = self.sortedFirstBySecond(index, first, reverse=reverse)
	return first, index


